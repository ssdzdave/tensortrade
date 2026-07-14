"""Append-only product state: the auditable track record.

Three flat, git-committed text files under ``state/``:

* ``signals.jsonl`` — one JSON object per (date, asset, voice) plus an
  ensemble row per asset per day; never rewritten, only appended.
* ``ledger.csv``    — every paper trade (plus the initial cash deposit).
* ``equity.csv``    — one NAV mark per day vs the blended buy-and-hold NAV.

All appends are idempotent: re-running the pipeline for a date that is
already recorded is a no-op, enforced here rather than by callers. Current
holdings are always *recomputed from the ledger* — there is no snapshot
file to drift out of sync.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SIGNALS_FILE = "signals.jsonl"
LEDGER_FILE = "ledger.csv"
EQUITY_FILE = "equity.csv"

LEDGER_COLUMNS = ["date", "asset", "side", "qty", "fill_price", "fee",
                  "cash_after", "note"]
EQUITY_COLUMNS = ["date", "nav", "cash", "positions_value", "buy_hold_nav"]

ENSEMBLE_VOICE = "ensemble"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _iso(day) -> str:
    if isinstance(day, (dt.date, dt.datetime)):
        return day.strftime("%Y-%m-%d")
    return str(day)


class StateStore:
    def __init__(self, state_dir) -> None:
        self.dir = Path(state_dir)
        self.signals_path = self.dir / SIGNALS_FILE
        self.ledger_path = self.dir / LEDGER_FILE
        self.equity_path = self.dir / EQUITY_FILE

    # -- signals ------------------------------------------------------------

    def read_signals(self) -> List[dict]:
        if not self.signals_path.exists():
            return []
        rows = []
        with open(self.signals_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def has_ensemble_rows(self, as_of) -> bool:
        day = _iso(as_of)
        return any(
            r["date"] == day and r["voice"] == ENSEMBLE_VOICE
            for r in self.read_signals()
        )

    def latest_stances(self) -> Dict[str, int]:
        """Most recent recorded ensemble stance per asset."""
        stances: Dict[str, int] = {}
        for row in self.read_signals():  # file is append-only => chronological
            if row["voice"] == ENSEMBLE_VOICE:
                stances[row["asset"]] = int(row["stance"])
        return stances

    def append_signals(self, as_of, voices_by_asset: Dict[str, list],
                       results: Dict[str, "EnsembleResult"]) -> int:
        """Append per-voice and ensemble rows for ``as_of``. Rows whose
        (date, asset, voice) already exist are skipped. Returns rows added."""
        day = _iso(as_of)
        existing = {(r["date"], r["asset"], r["voice"]) for r in self.read_signals()}
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        sha = git_sha()

        added = 0
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.signals_path, "a") as fh:
            for asset, voices in voices_by_asset.items():
                for voice in voices:
                    key = (day, asset, voice.name)
                    if key in existing:
                        continue
                    fh.write(json.dumps({
                        "date": day,
                        "asset": asset,
                        "voice": voice.name,
                        "stance": int(voice.stance),
                        "weight": voice.weight,
                        "available": voice.available,
                        "rationale": voice.rationale,
                        "generated_at_utc": stamp,
                        "git_sha": sha,
                    }) + "\n")
                    added += 1
                result = results.get(asset)
                key = (day, asset, ENSEMBLE_VOICE)
                if result is not None and key not in existing:
                    fh.write(json.dumps({
                        "date": day,
                        "asset": asset,
                        "voice": ENSEMBLE_VOICE,
                        "stance": int(result.stance),
                        "score": round(result.score, 6),
                        "n_long": result.n_long,
                        "n_voices": result.n_voices,
                        "prev_stance": result.prev_stance,
                        "changed": result.changed,
                        "rationale": result.vote_summary(),
                        "generated_at_utc": stamp,
                        "git_sha": sha,
                    }) + "\n")
                    added += 1
        return added

    # -- ledger ---------------------------------------------------------------

    def read_ledger(self) -> pd.DataFrame:
        if not self.ledger_path.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        return pd.read_csv(self.ledger_path)

    def append_ledger_rows(self, rows: List[dict]) -> int:
        """Append trades, skipping any (date, asset) pair already recorded."""
        if not rows:
            return 0
        ledger = self.read_ledger()
        existing = {(str(r["date"]), str(r["asset"])) for _, r in ledger.iterrows()}
        fresh = [r for r in rows if (str(r["date"]), str(r["asset"])) not in existing]
        if not fresh:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(fresh, columns=LEDGER_COLUMNS)
        frame.to_csv(self.ledger_path, mode="a", index=False,
                     header=not self.ledger_path.exists())
        return len(fresh)

    def holdings(self) -> Tuple[float, Dict[str, float]]:
        """(cash, {symbol: qty}) recomputed from the ledger."""
        cash = 0.0
        positions: Dict[str, float] = {}
        for _, row in self.read_ledger().iterrows():
            side = row["side"]
            qty = float(row["qty"])
            price = float(row["fill_price"])
            fee = float(row["fee"])
            symbol = str(row["asset"])
            if side == "DEPOSIT":
                cash += qty
            elif side == "BUY":
                cash -= qty * price + fee
                positions[symbol] = positions.get(symbol, 0.0) + qty
            elif side == "SELL":
                cash += qty * price - fee
                positions[symbol] = positions.get(symbol, 0.0) - qty
        positions = {s: q for s, q in positions.items() if abs(q) > 1e-12}
        return cash, positions

    # -- equity ---------------------------------------------------------------

    def read_equity(self) -> pd.DataFrame:
        if not self.equity_path.exists():
            return pd.DataFrame(columns=EQUITY_COLUMNS)
        df = pd.read_csv(self.equity_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def append_equity_row(self, row: dict) -> bool:
        """Append one NAV mark; a row for the same date is a no-op."""
        day = _iso(row["date"])
        equity = self.read_equity()
        if not equity.empty and (equity["date"].dt.strftime("%Y-%m-%d") == day).any():
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        out = {**row, "date": day}
        pd.DataFrame([out], columns=EQUITY_COLUMNS).to_csv(
            self.equity_path, mode="a", index=False,
            header=not self.equity_path.exists(),
        )
        return True

    def inception(self) -> Optional[pd.Timestamp]:
        equity = self.read_equity()
        if equity.empty:
            return None
        return equity["date"].iloc[0]
