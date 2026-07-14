"""Email delivery via stdlib smtplib — no provider SDK needed.

Recipients are always BCC'd (passed as envelope recipients, never written
into headers) so clients don't see each other. ``dry_run`` writes the fully
rendered ``.eml`` to the outbox for inspection instead of sending.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from typing import Optional

CHART_CID = "equity_chart"


def build_message(cfg, artifacts) -> EmailMessage:
    msg = EmailMessage()
    subject = f"{cfg.email.subject_prefix} {artifacts.email_subject}".strip()
    msg["Subject"] = subject
    msg["From"] = cfg.email.from_addr
    msg["To"] = cfg.email.from_addr  # real recipients ride the envelope (BCC)
    msg["Date"] = formatdate(localtime=False, usegmt=True)
    msg.set_content(artifacts.email_text)
    msg.add_alternative(artifacts.email_html, subtype="html")
    if artifacts.chart_png:
        html_part = msg.get_payload()[1]
        html_part.add_related(
            artifacts.chart_png, maintype="image", subtype="png",
            cid=f"<{CHART_CID}>",
        )
    return msg


def send_email(cfg, artifacts, dry_run: bool = False) -> Optional[Path]:
    """Send (or, on dry_run, write to the outbox). Returns the outbox path
    when a file was written, else None."""
    msg = build_message(cfg, artifacts)

    if dry_run:
        outbox = Path(cfg.report.outbox_dir)
        outbox.mkdir(parents=True, exist_ok=True)
        path = outbox / f"{artifacts.as_of}.eml"
        path.write_bytes(bytes(msg))
        return path

    if not cfg.email.bcc:
        raise ValueError("email.bcc is empty — nobody to send to")
    username = cfg.email.username or cfg.email.from_addr
    recipients = [cfg.email.from_addr, *cfg.email.bcc]
    with smtplib.SMTP(cfg.email.smtp_host, cfg.email.smtp_port, timeout=60) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(username, cfg.email.password())
        smtp.send_message(msg, from_addr=cfg.email.from_addr, to_addrs=recipients)
    return None
