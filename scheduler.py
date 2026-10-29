"""
Daily scheduler for the fraud-detection platform.

Runs as a long-lived process (or Windows Task / cron one-shot) and executes:
  1. api/build_graph_cache.py  — rebuild GNN serving cache from latest model
  2. monitoring/drift_detection.py — PSI/KS drift report
  3. monitoring/alerting.py    — email/Slack notification if drift alerts fire

Usage:
    python scheduler.py            # loop forever, daily at RUN_AT_HOUR
    python scheduler.py --once     # run the whole pipeline once and exit
"""
import argparse
import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scheduler")

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STEPS = [
    ("graph-cache", [PYTHON, str(ROOT / "api" / "build_graph_cache.py")]),
    ("drift-detection", [PYTHON, str(ROOT / "monitoring" / "drift_detection.py")]),
    ("alerting", [PYTHON, str(ROOT / "monitoring" / "alerting.py")]),
]


def run_pipeline():
    for name, cmd in STEPS:
        log.info("=== STEP %s ===", name)
        try:
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                    timeout=3600)
            if result.returncode == 0:
                log.info("[%s] OK", name)
            else:
                log.error("[%s] FAILED (exit %s): %s",
                          name, result.returncode, result.stderr[-500:])
        except Exception as e:
            log.error("[%s] crashed: %s", name, e)


def seconds_until_next_run(hour: int) -> float:
    now = datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def main():
    parser = argparse.ArgumentParser(description="Daily job scheduler")
    parser.add_argument("--once", action="store_true",
                        help="Run the pipeline once and exit")
    parser.add_argument("--hour", type=int, default=2,
                        help="Hour of day (0-23) to run daily jobs")
    args = parser.parse_args()

    if args.once:
        log.info("One-shot pipeline run")
        run_pipeline()
        return

    log.info("Scheduler started — daily at %02d:00", args.hour)
    while True:
        wait = seconds_until_next_run(args.hour)
        log.info("Next run in %.1f hours", wait / 3600)
        time.sleep(wait)
        run_pipeline()


if __name__ == "__main__":
    main()
