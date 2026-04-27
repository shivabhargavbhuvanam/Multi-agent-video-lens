"""
Demo recorder for Video Lens.
Records a walkthrough of all three screens: Welcome → Loading → Chat.
Saves demo.webm then converts to demo.mp4 via ffmpeg.
"""

import subprocess, time, os, glob
from playwright.sync_api import sync_playwright

DEMO_URL   = "https://www.youtube.com/watch?v=aQQUl22NNGE"
SERVER     = "http://127.0.0.1:5000"
VIDEO_DIR  = "./demo_tmp"
OUT_FILE   = "demo.mp4"
W, H       = 1280, 720


def record():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            record_video_dir=VIDEO_DIR,
            record_video_size={"width": W, "height": H},
            viewport={"width": W, "height": H},
        )
        page = ctx.new_page()

        # ── 1. Welcome screen ─────────────────────────────────────
        print("Recording: Welcome screen…")
        page.goto(SERVER)
        page.wait_for_selector(".welcome-title")
        page.wait_for_timeout(2500)

        # ── 2. Type URL ───────────────────────────────────────────
        print("Recording: Typing URL…")
        box = page.locator("#url-input")
        box.click()
        page.wait_for_timeout(500)
        for ch in DEMO_URL:
            page.keyboard.type(ch, delay=18)
        page.wait_for_timeout(1500)

        # ── 3. Transition to loading (skip real processing) ───────
        print("Recording: Loading screen…")
        page.evaluate("show(document.getElementById('loading-screen'))")
        page.wait_for_timeout(800)

        steps = ["step-download", "step-process", "step-yolo", "step-embed"]
        for step in steps:
            page.evaluate(f"""
                document.getElementById('{step}').classList.add('active');
                document.getElementById('loading-detail').textContent = document.getElementById('{step}').innerText.trim();
            """)
            page.wait_for_timeout(1200)
            page.evaluate(f"""
                document.getElementById('{step}').classList.remove('active');
                document.getElementById('{step}').classList.add('done');
            """)
            page.wait_for_timeout(400)

        page.wait_for_timeout(800)

        # ── 4. Transition to chat ─────────────────────────────────
        print("Recording: Chat screen…")
        page.evaluate(f"openChat('{DEMO_URL}')")
        page.wait_for_timeout(2000)

        # ── 5. Ask a question and show real answer ────────────────
        print("Recording: Asking question…")
        chat_input = page.locator("#chat-input")
        chat_input.click()
        question = "What objects appear in the video?"
        for ch in question:
            page.keyboard.type(ch, delay=22)
        page.wait_for_timeout(1000)
        page.keyboard.press("Enter")

        # Wait for response (real API call)
        page.wait_for_timeout(12000)

        # ── 6. Ask a second question ──────────────────────────────
        print("Recording: Asking second question…")
        chat_input.click()
        q2 = "What is being said in the video?"
        for ch in q2:
            page.keyboard.type(ch, delay=22)
        page.wait_for_timeout(800)
        page.keyboard.press("Enter")
        page.wait_for_timeout(12000)

        # ── 7. Hold on chat for a moment ──────────────────────────
        page.wait_for_timeout(2500)

        print("Saving recording…")
        ctx.close()
        browser.close()

    # Find the recorded .webm
    webm_files = glob.glob(os.path.join(VIDEO_DIR, "*.webm"))
    if not webm_files:
        print("ERROR: No .webm file found.")
        return

    webm = webm_files[0]
    print(f"Converting {webm} → {OUT_FILE}…")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", webm,
        "-vf", f"scale={W}:{H}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-movflags", "+faststart",
        OUT_FILE,
    ], check=True)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(VIDEO_DIR, ignore_errors=True)
    print(f"✓ Demo saved to: {OUT_FILE}")


if __name__ == "__main__":
    record()
