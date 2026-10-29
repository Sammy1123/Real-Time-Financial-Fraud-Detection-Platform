"""
Automated dashboard demo capture.

Opens each Streamlit page in a headless browser, screenshots it, and (with
imageio available) stitches the frames into demo.gif for the README.

Usage:
    1. Start the dashboard:  python main.py dashboard
    2. Run this script:      py -3.10 data/capture_demo.py
    3. Optional GIF stitch:  py -3.10 -m pip install imageio pillow
                             py -3.10 data/capture_demo.py --gif

Frames are saved to docs/demo_frames/*.png; commit the final GIF as
docs/demo.gif and reference it in README.md.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "docs" / "demo_frames"
FRAMES.mkdir(parents=True, exist_ok=True)

# (page radio label, png filename)
PAGES = [
    ("Overview",          "01_overview.png"),
    ("Transactions",      "02_transactions.png"),
    ("Model Performance", "03_model_performance.png"),
    ("Live Alerts",       "04_live_alerts.png"),
    ("Drift Monitor",     "05_drift_monitor.png"),
]
BASE_URL = "http://localhost:8501"


def capture():
    """Drive the running dashboard via its WebSocket protocol using playwright
    if installed, falling back to per-page URL screenshots otherwise."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — install with:")
        print("  py -3.10 -m pip install playwright")
        print("  py -3.10 -m playwright install chromium")
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        page.goto(BASE_URL, wait_until="networkidle")
        time.sleep(3)  # initial render

        for label, filename in PAGES:
            # click the sidebar radio for this page
            try:
                page.get_by_text(label, exact=True).first.click()
            except Exception:
                # Streamlit renders radio options as labels; fall back
                page.locator(f"label:has-text('{label}')").first.click()
            time.sleep(6)  # let cached charts render
            page.screenshot(path=str(FRAMES / filename), full_page=False)
            print(f"captured {filename}")
        browser.close()
    return True


def stitch_gif():
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not installed — skipping GIF stitch "
              "(pip install imageio pillow)")
        return
    frames = [imageio.imread(f) for f in sorted(FRAMES.glob("*.png"))]
    out = ROOT / "docs" / "demo.gif"
    imageio.mimsave(out, frames, duration=2.5)
    print(f"GIF written to {out} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture dashboard demo frames")
    parser.add_argument("--gif", action="store_true",
                        help="Also stitch captured frames into docs/demo.gif")
    args = parser.parse_args()

    if not capture():
        sys.exit(1)
    if args.gif:
        stitch_gif()
