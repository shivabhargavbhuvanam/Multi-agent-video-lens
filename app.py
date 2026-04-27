import threading
import asyncio
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Global state ───────────────────────────────────────────────────────────────

_state = {
    "status": "idle",   # idle | validating | downloading | processing | ready | error
    "message": "",
    "video_url": None,
}
_lock = threading.Lock()


def _set(status, message=""):
    with _lock:
        _state["status"] = status
        _state["message"] = message


def _get():
    with _lock:
        return dict(_state)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _validate_url(url: str):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return True, info.get("title", "Video")
    except Exception as exc:
        return False, str(exc)


def _run_pipeline(url: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        _set("validating", "Checking video URL…")
        ok, title_or_err = _validate_url(url)
        if not ok:
            _set("error", f"Could not access video: {title_or_err}")
            return

        _set("downloading", f'Downloading "{title_or_err}"…')
        from hackathon import (
            download_video, VideoProcessingWorkflow,
            create_yolo_chunks, generate_embedding,
        )

        video_id = "uploaded_video"
        source = f"sources/{video_id}/"
        video_path = source + "movie.mp4"
        output_dir = source + "chunks"

        download_video(url, source)

        _set("processing", "Processing video — extracting frames, audio & transcripts…")
        wf = VideoProcessingWorkflow(video_path, output_dir)
        loop.run_until_complete(wf.split_video_and_audio())

        _set("processing", "Running YOLO object detection…")
        create_yolo_chunks()

        _set("processing", "Generating vector embeddings…")
        generate_embedding("image_captioning", video_id, source)
        generate_embedding("transcripts", video_id, source)

        _set("ready", "Analysis complete!")
    except Exception as exc:
        _set("error", str(exc))
    finally:
        loop.close()


# ── Query helper (lazy import so startup stays fast) ──────────────────────────

def _query_video(question: str) -> str:
    from workflow import image_captioning_engine, transcripts_engine, yolo_engine, gpt

    route_resp = gpt.complete(
        "You are a routing agent for a video analysis system.\n"
        "Pick exactly one route based on the question:\n"
        "- transcripts: speech, dialogue, audio, what was said, names mentioned\n"
        "- image_captioning: visual scenes, colors, appearance, people, actions\n"
        "- yolo: only when the user explicitly asks about object detection\n\n"
        f"Question: {question}\n\n"
        "Reply with exactly one word: transcripts, image_captioning, or yolo"
    )
    route = str(route_resp).strip().lower()

    if "transcript" in route:
        return str(transcripts_engine.query(question))
    if "yolo" in route:
        return str(yolo_engine.query(question))
    return str(image_captioning_engine.query(question))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit-url", methods=["POST"])
def submit_url():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    with _lock:
        if _state["status"] not in ("idle", "ready", "error"):
            return jsonify({"error": "Already processing a video"}), 409
        _state["video_url"] = url
        _state["status"] = "starting"

    threading.Thread(target=_run_pipeline, args=(url,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    return jsonify(_get())


@app.route("/chat", methods=["POST"])
def chat():
    if _get()["status"] != "ready":
        return jsonify({"error": "Video not ready yet"}), 400

    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = _query_video(question)
        return jsonify({"answer": answer})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    _set("idle", "")
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
