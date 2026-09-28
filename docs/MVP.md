# Daily Crypto Signals — the MVP product layer

`daily_signals/` turns this research repo into a sellable product: an
automated **daily trading brief** (email + web report) for a select client
list, backed by a transparent strategy ensemble, an honest backtest, and a
git-committed forward track record.

---

## 1. Why this shape — the repo assessment

TensorTrade (this fork) is a strong **research core** and was missing
**everything product-shaped**:

| Had | Missing (now provided by `daily_signals/`) |
|---|---|
| Simulated OMS (orders, wallets, commissions, slippage) | Live/recent data (only static historical CSVs existed) |
| Composable RL env (BSH actions, PBR rewards) | Inference path (training checkpoints went to `/tmp` and were deleted) |
| Ray RLlib training + Optuna tuning + walk-forward docs | Scheduling, delivery (email/web), persistence across runs |
| Honest experiment log (`docs/EXPERIMENTS.md`) | Performance reporting (Sharpe existed only as an RL reward) |
| 250+ tests, CI | Multi-asset support, CLI, any user-facing surface |

**The cadence decision follows the repo's own research.** The hourly RL agent
predicted direction (+$239 vs buy-and-hold's −$355 at zero commission) but
overtraded (~2,000 trades/30 days), so fees erased the edge. The documented
fix is trading ~10× less. A **daily** product — stateful stances, positions
held days-to-weeks, single-digit trades per month — is exactly that regime,
and it needs no intraday monitoring.

## 2. Architecture

```
 Coinbase public REST ──┐  (ccxt fallback)
                        ▼
                data/cache/*.csv          incremental, validated, atomic
                        ▼
     ┌──────────────────┼──────────────────────┐
     ▼                  ▼                      ▼
 trend_sma        momentum_roc           meanrev_rsi     (+ optional rl_ppo)
     └──────────────────┼──────────────────────┘
                        ▼
              weighted ensemble vote            score > 0.5 → LONG, tie → keep
                        ▼
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
 state/signals.jsonl  state/ledger.csv  state/equity.csv    ← committed daily
        └───────────────┼────────────────┘
                        ▼
        reports/site/index.html  +  email (SMTP, BCC)
```

Everything above runs on the **light install** (`pandas numpy requests
jinja2 pyyaml matplotlib`) — no tensorflow/ray/torch. The RL voice is an
optional plug-in (§7): if its checkpoint or ray is absent, it is excluded
from the vote and the report says "unavailable"; the pipeline never fails
because of it.

### The signal model

* Each **voice** maps daily OHLCV to a stance series (1 = LONG, 0 = FLAT)
  where row *t* uses only data through close(*t*). The same series backs the
  backtest and the daily signal (its last value) — what clients receive is
  exactly what was tested.
* The **ensemble** is a weighted vote over available voices. An exact tie
  keeps yesterday's stance, damping flip-flops (the overtrading failure mode
  from the experiments).
* **Execution convention** (backtest AND paper ledger): stances decided at
  the UTC close fill at that close — in a 24/7 market the price seconds
  after 00:00 UTC is the close; every position change pays
  `commission_bps` (default 15 bps/side) to cover fees + slippage.

## 3. "Simulate it working" — the two evidence layers

1. **Historical**: `daily-signals backtest` prints the tearsheet (CAGR,
   Sharpe, Sortino, max drawdown, win rate, trades/month, exposure, vs
   buy-and-hold) per voice, per asset, plus the blended portfolio — same
   engine, same costs, no lookahead (`pos = stance.shift(1)`, verified by
   test).
2. **Forward**: every daily run appends to `state/signals.jsonl`,
   `state/ledger.csv` and `state/equity.csv` and **commits them**. The git
   history is the tamper-evident audit trail: each signal exists in a public
   commit *before* the next bar happens. Run it 4–8 weeks and you have an
   out-of-sample track record to show clients (`git log --follow
   state/signals.jsonl`).

## 4. Runbook

```bash
# light install (no tensorflow/ray)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r daily_signals/requirements.txt
pip install -e . --no-deps          # optional: gives the daily-signals command

daily-signals init                  # copies example config, fetches history, backtests
$EDITOR config/signals.yaml         # assets, weights, strategies, email
daily-signals backtest              # tearsheet to stdout
daily-signals run-daily --dry-run   # full pipeline; email lands in reports/outbox/
open reports/site/index.html

# real email (BCC yourself first)
export SIGNALS_SMTP_PASSWORD='app-password'
daily-signals send-email
```

Subcommands: `fetch` (refresh cache), `backtest`, `run-daily [--dry-run]
[--resend]`, `render` (rebuild reports from state, no fetch/append),
`send-email [--dry-run]`, `init`. Config: `config/signals.yaml` (see the
commented example). Secrets only via environment (`SIGNALS_SMTP_PASSWORD`).

Re-running `run-daily` on an already-recorded day re-renders reports but
appends nothing and does not re-email (unless `--resend`) — safe to retry.
If data cannot be fetched the run falls back to the cache, sends a
prominent **STALE DATA** banner instead of a signal, records nothing, and
exits 0.

## 5. Automation (GitHub Actions)

`.github/workflows/daily-signals.yml` runs at **00:15 UTC** daily (and on
manual dispatch): light install → `run-daily` → commit `state/` +
`reports/site/` → deploy the site to GitHub Pages.

One-time setup:

1. **Secret**: repo → Settings → Secrets and variables → Actions → add
   `SIGNALS_SMTP_PASSWORD` (for Gmail: an app password; the SMTP host/port/
   from/bcc live in `config/signals.yaml`).
2. **Pages**: repo → Settings → Pages → Source: **GitHub Actions**. Put the
   resulting URL into `report.site_url` so emails link to it.
3. Commit your `config/signals.yaml` (it contains no secrets). Until it
   exists the workflow uses the example config with email disabled — the
   paper track record still accrues.
4. First run: Actions → Daily Signals → **Run workflow**.

Local alternative: `crontab -e` →
`15 0 * * * cd /path/to/repo && .venv/bin/python -m daily_signals run-daily`.

## 6. Email deliverability

Stdlib SMTP is fine for a select client list. Gmail: enable 2FA, create an
App Password, `smtp.gmail.com:587`, set `from_addr` to your address and put
clients in `bcc` (they are envelope recipients only — never visible to each
other). If you outgrow it (volume, DKIM/SPF control), swap `emailer.py` for
Resend/SendGrid — it is one function.

## 7. The RL voice (optional differentiator)

```bash
pip install -e ".[rl]"                                   # ray[rllib] + torch<2.5
python -m daily_signals.rl.train_daily --asset BTC-USD   # ~10-20 min/asset
```

(The torch cap matters: ray 2.37's RLlib segfaults under the much newer
torch releases; torch 2.4.x is the verified pairing.)

Training adapts the tuned pipeline from `examples/training/train_best.py`
to daily bars from the same cache the signals use, and persists the best
checkpoint to `models/BTC-USD/ppo/` with a `manifest.json` (features,
window, validation P&L, git sha) — nothing goes to `/tmp`. Set
`rl.enabled: true` in the config and the PPO agent becomes one more voice:
its stance history replays through the same backtester into the tearsheet,
so its contribution is measured, not asserted. Keep expectations calibrated
by `docs/EXPERIMENTS.md`: the agent's edge is real but thin — that is why
it is one voice among several, not the product.

## 8. Extending

* **Add an asset**: one line in `assets:` (any Coinbase product id, e.g.
  `{symbol: LINK-USD, weight: 0.1}`); weights must sum to ≤ 1.0.
* **Add a strategy**: subclass `Strategy` in `daily_signals/strategies/`,
  register it in `REGISTRY` and `DEFAULT_STRATEGY_PARAMS`, add a test with
  hand-known transitions. It automatically appears in the vote, reports and
  tearsheet.
* **Equities later**: implement `OHLCVSource.fetch()` for your vendor
  (yfinance/polygon) behind `data.source`; the rest of the pipeline is
  cadence- and venue-agnostic (mind trading calendars: `ppy=252`).

## 9. Honesty notes (what to tell clients)

* The email/site disclaimer ships on every artifact: research output, not
  investment advice. Selling subscription research generally keeps you
  clearer of advisory registration than managing money — confirm with a
  lawyer for your jurisdiction.
* Backtests are simulated, fills are idealized (close ± costs), and the
  synthetic fixtures in tests are for plumbing, not performance claims.
  Quote the committed forward track record, not just the backtest.
* Sharpe/CAGR here use daily bars and 365 periods/year; they are not
  directly comparable to equity-market numbers computed on 252.
