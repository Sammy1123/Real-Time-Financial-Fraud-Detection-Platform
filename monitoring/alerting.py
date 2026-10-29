"""
Drift alerting — notifies on drift-report alerts.

Reads monitoring/drift_report.json and, when data_drift_alert or
model_drift_alert is True, sends notifications to every channel that is
configured in .env:

  ALERT_SLACK_WEBHOOK   Slack incoming-webhook URL
  ALERT_EMAIL_TO        comma-separated recipient list
  ALERT_EMAIL_FROM      sender address          (required for email)
  SMTP_HOST / SMTP_PORT SMTP server              (default localhost:25)
  SMTP_USER / SMTP_PASS optional SMTP auth

Channels that are not configured are skipped with a log line, so the
script never fails just because email isn't set up.
"""
import json
import logging
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

REPORT  = Path("monitoring") / "drift_report.json"
LOGS    = Path("logs")
LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOGS / "alerting.log"), logging.StreamHandler()],
)
log = logging.getLogger("alerting")


def _format_message(report: dict) -> tuple[str, str]:
    """Return (subject, text_body) for the alert."""
    alerts = report.get("alerts", {})
    md     = report.get("model_drift", {})
    dd     = report.get("data_drift", {})

    fired = [k for k, v in alerts.items() if v]
    subject = "[fraud-platform] DRIFT ALERT: " + ", ".join(
        k.replace("_alert", "").replace("_drift", " drift") for k in fired)

    lines = [
        f"Drift detection run — {report.get('run_timestamp', 'unknown')}",
        f"Reference rows: {report.get('reference_size', '?'):,}  "
        f"Current rows: {report.get('current_size', '?'):,}",
        "",
        f"Alerts fired: {', '.join(fired) or 'none'}",
        "",
        "Model drift:",
        f"  score PSI        : {md.get('score_psi')}",
        f"  score KS p-value : {md.get('score_ks_pvalue')}",
        f"  fraud rate delta : {md.get('fraud_rate_delta', 0):+.4%}",
        "",
        "Data drift (excluding identity-only M-features):",
        f"  features drifted : {dd.get('features_drifted', 0)}",
        f"  features warned  : {dd.get('features_warned', 0)}",
        f"  excluded (coverage skew): {dd.get('excluded_identity_only', [])}",
    ]
    top = dd.get("top_drifted_features") or {}
    if top:
        lines.append("  top drifted      :")
        lines += [f"    {k}: {v:.4f}" for k, v in list(top.items())[:5]]

    return subject, "\n".join(lines)


def send_slack(subject: str, body: str) -> bool:
    webhook = os.getenv("ALERT_SLACK_WEBHOOK")
    if not webhook:
        log.info("Slack webhook not configured — skipping")
        return False
    try:
        resp = requests.post(webhook, json={"text": f"*{subject}*\n```{body}```"},
                             timeout=10)
        resp.raise_for_status()
        log.info("Slack alert sent")
        return True
    except Exception as e:
        log.error("Slack send failed: %s", e)
        return False


def send_email(subject: str, body: str) -> bool:
    to = os.getenv("ALERT_EMAIL_TO")
    frm = os.getenv("ALERT_EMAIL_FROM")
    if not to or not frm:
        log.info("Email not configured (ALERT_EMAIL_TO / ALERT_EMAIL_FROM) — skipping")
        return False

    host = os.getenv("SMTP_HOST", "localhost")
    port = int(os.getenv("SMTP_PORT", "25"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = frm
    msg["To"] = to

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port,
                                      context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(host, port)
        with server:
            if user and password:
                server.starttls()
                server.login(user, password)
            server.sendmail(frm, [a.strip() for a in to.split(",")], msg.as_string())
        log.info("Email alert sent to %s", to)
        return True
    except Exception as e:
        log.error("Email send failed: %s", e)
        return False


def main():
    if not REPORT.exists():
        log.error("No drift report at %s — run drift_detection.py first", REPORT)
        return

    report = json.load(open(REPORT))
    alerts = report.get("alerts", {})
    fired  = [k for k, v in alerts.items() if v]

    if not fired:
        log.info("No drift alerts — nothing to send")
        return

    subject, body = _format_message(report)
    log.info("Drift alerts fired: %s", ", ".join(fired))
    print(body)

    sent = send_slack(subject, body)
    sent |= send_email(subject, body)
    if not sent:
        log.warning("Alerts fired but no channel was configured — "
                    "set ALERT_SLACK_WEBHOOK or ALERT_EMAIL_TO in .env")


if __name__ == "__main__":
    main()
