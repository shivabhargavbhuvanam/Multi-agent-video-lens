"""
Demo recorder for Video Lens.
Records a walkthrough: Welcome → URL input → Loading → Chat with Q&A.
Saves as demo.mp4 via ffmpeg.
"""

import subprocess, os, glob, shutil
from playwright.sync_api import sync_playwright

DEMO_URL  = "https://www.youtube.com/watch?v=dxKPCPMaWFg"
SERVER    = "http://127.0.0.1:5000"
VIDEO_DIR = "./demo_tmp"
OUT_FILE  = "demo.mp4"
W, H      = 1280, 720


def type_naturally(page, text, delay=40):
    """Type text character by character at a human-like pace."""
    for ch in text:
        page.keyboard.type(ch, delay=delay)


def record():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--force-device-scale-factor=1"],
        )
        ctx = browser.new_context(
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": W, "height": H},
            viewport={"width": W, "height": H},
        )
        page = ctx.new_page()

        # ── 1. Welcome screen — hold for a beat ───────────────────
        print("▶  Welcome screen")
        page.goto(SERVER)
        page.wait_for_selector(".welcome-title", timeout=10000)
        page.wait_for_timeout(3000)          # viewer reads the title

        # ── 2. Click the input, type URL slowly ───────────────────
        print("▶  Typing URL")
        url_box = page.locator("#url-input")
        url_box.click()
        page.wait_for_timeout(600)
        type_naturally(page, DEMO_URL, delay=45)
        page.wait_for_timeout(1200)          # pause before clicking Analyze

        # ── 3. Click Analyze button visibly ───────────────────────
        print("▶  Clicking Analyze")
        page.locator("#submit-btn").click()
        page.wait_for_timeout(800)

        # ── 4. Switch to loading screen (bypass real processing) ──
        print("▶  Loading screen")
        page.evaluate("show(document.getElementById('loading-screen'))")
        page.wait_for_timeout(1000)

        step_labels = {
            "step-download": "Downloading video…",
            "step-process":  "Extracting frames & transcripts…",
            "step-yolo":     "Running object detection…",
            "step-embed":    "Building vector embeddings…",
        }

        for step_id, label in step_labels.items():
            # Activate current step
            page.evaluate(f"""
                document.getElementById('{step_id}').classList.add('active');
                document.getElementById('loading-detail').textContent = '{label}';
            """)
            page.wait_for_timeout(1600)
            # Mark done, move to next
            page.evaluate(f"""
                document.getElementById('{step_id}').classList.remove('active');
                document.getElementById('{step_id}').classList.add('done');
            """)
            page.wait_for_timeout(300)

        page.evaluate("document.getElementById('loading-detail').textContent = 'Analysis complete!';")
        page.wait_for_timeout(1200)

        # ── 5. Open chat screen ───────────────────────────────────
        print("▶  Chat screen")
        page.evaluate(f"openChat('{DEMO_URL}')")
        page.wait_for_timeout(2500)          # viewer reads the greeting

        # ── 6. First question ─────────────────────────────────────
        print("▶  Question 1")
        chat = page.locator("#chat-input")
        chat.click()
        page.wait_for_timeout(400)
        type_naturally(page, "What objects appear in the video?", delay=38)
        page.wait_for_timeout(900)
        page.keyboard.press("Enter")

        # Wait for typing dots then answer
        page.wait_for_selector("#typing", timeout=8000)
        page.wait_for_selector("#typing", state="detached", timeout=30000)
        page.wait_for_timeout(2500)

        # ── 7. Second question ────────────────────────────────────
        print("▶  Question 2")
        chat.click()
        page.wait_for_timeout(400)
        type_naturally(page, "What is being said in the video?", delay=38)
        page.wait_for_timeout(900)
        page.keyboard.press("Enter")

        page.wait_for_selector("#typing", timeout=8000)
        page.wait_for_selector("#typing", state="detached", timeout=30000)
        page.wait_for_timeout(3000)          # hold on final answer

        # ── 8. Save ───────────────────────────────────────────────
        print("▶  Saving…")
        ctx.close()
        browser.close()

    # Convert .webm → .mp4
    webm_files = glob.glob(os.path.join(VIDEO_DIR, "*.webm"))
    if not webm_files:
        print("ERROR: No .webm found.")
        return

    webm = webm_files[0]
    print(f"Converting {webm} → {OUT_FILE}")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", webm,
        "-vf", f"scale={W}:{H}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        OUT_FILE,
    ], check=True, capture_output=True)

    shutil.rmtree(VIDEO_DIR, ignore_errors=True)
    size_mb = os.path.getsize(OUT_FILE) / 1_000_000
    print(f"✓ demo.mp4 saved ({size_mb:.1f} MB)")


if __name__ == "__main__":
    record()
