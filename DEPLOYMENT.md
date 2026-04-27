# Deployment Guide — Video Lens

## Recommended Platform: Railway

Railway is the best fit for this project because it:
- Deploys directly from your GitHub repo with zero config
- Supports Docker (needed for `ffmpeg` and system-level dependencies)
- Handles environment variables securely
- Auto-restarts on crash
- Scales easily if traffic grows

For a production-grade setup with heavier video workloads, **AWS EC2** (c5.xlarge or better) is the alternative — more control, persistent disk, and optional GPU support.

---

## Prerequisites

Before deploying, make sure you have:
- [ ] A [Railway](https://railway.app) account (free tier works to start)
- [ ] Your GitHub repo pushed and up to date
- [ ] All three API keys ready:
  - `OPENAI_API_KEY`
  - `PINECONE_API_KEY`
  - `HF_TOKEN`

---

## Step 1 — Add a Dockerfile

Railway needs a Dockerfile to install system dependencies (`ffmpeg`, Python packages).

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
```

---

## Step 2 — Update app.py for Production Port

Railway injects a `PORT` environment variable. Update the last line of `app.py`:

```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```

---

## Step 3 — Add a requirements update

Add `gunicorn` to `requirements.txt` for a production-grade server:

```
gunicorn
flask
```

Update the Dockerfile CMD to use gunicorn:

```dockerfile
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "app:app"]
```

> **Important:** Use `--workers 1` — the pipeline uses in-process global state. Multiple workers would cause race conditions.

---

## Step 4 — Add a .dockerignore

Create `.dockerignore` in the project root to keep the image lean:

```
venv/
__pycache__/
*.pyc
sources/
*.mp4
*.mp3
*.jpg
*.png
.env
.git/
```

---

## Step 5 — Deploy on Railway

1. Go to [railway.app](https://railway.app) and click **New Project**
2. Select **Deploy from GitHub repo**
3. Choose `shivabhargavbhuvanam/Multi-agent-video-lens`
4. Railway will auto-detect the Dockerfile and start building

---

## Step 6 — Set Environment Variables

In the Railway dashboard:
1. Click your service → **Variables** tab
2. Add the following:

| Variable | Value |
|---|---|
| `OPENAI_API_KEY` | your OpenAI key |
| `PINECONE_API_KEY` | your Pinecone key |
| `HF_TOKEN` | your HuggingFace token |
| `IMAGEIO_FFMPEG_EXE` | `/usr/bin/ffmpeg` |

---

## Step 7 — Add Persistent Storage (Important)

Video processing writes files to `sources/`. Without persistent storage these are lost on every redeploy.

In Railway:
1. Click your service → **Volumes** tab
2. Click **Add Volume**
3. Mount path: `/app/sources`
4. This disk persists across deployments and restarts

---

## Step 8 — Get Your Public URL

After the build completes:
1. Railway dashboard → **Settings** → **Domains**
2. Click **Generate Domain** — you'll get a URL like `https://video-lens-production.up.railway.app`
3. Share this URL — your app is live

---

## Verifying the Deployment

Visit your Railway URL. You should see the Video Lens welcome screen. Test with a short YouTube video (under 5 minutes) for the first run to verify the full pipeline works end-to-end.

If the `/chat` endpoint times out, increase the gunicorn timeout:
```
--timeout 900
```

---

## Cost Estimate (Railway)

| Plan | vCPU | RAM | Price | Suitable for |
|---|---|---|---|---|
| Hobby | Shared | 512 MB | ~$5/mo | Light testing only |
| Pro (Starter) | 2 vCPU | 2 GB | ~$20/mo | Demo / portfolio |
| Pro (Performance) | 8 vCPU | 8 GB | ~$60/mo | Real workloads |

Video processing is CPU-intensive — the Pro (Starter) plan is the minimum for real use.

---

## Alternative: AWS EC2

For heavier workloads or full control:

1. Launch an EC2 `c5.xlarge` instance (4 vCPU, 8 GB RAM) with Ubuntu 22.04
2. Install dependencies:
   ```bash
   sudo apt update && sudo apt install -y ffmpeg python3-pip python3-venv
   ```
3. Clone your repo, set up venv, install requirements
4. Set environment variables in `/etc/environment`
5. Run with gunicorn behind nginx:
   ```bash
   gunicorn --bind 0.0.0.0:8000 --timeout 600 --workers 1 app:app
   ```
6. Use a domain + Let's Encrypt SSL via Certbot

EC2 cost: ~$70–120/mo for a `c5.xlarge` on-demand (use a Reserved Instance for ~40% savings).

---

## Summary Checklist

- [ ] Dockerfile created
- [ ] `app.py` updated to use `PORT` env var and `host="0.0.0.0"`
- [ ] `gunicorn` added to `requirements.txt`
- [ ] `.dockerignore` created
- [ ] GitHub repo is up to date
- [ ] Railway project created and linked to repo
- [ ] All 3 API keys set in Railway Variables
- [ ] Persistent volume mounted at `/app/sources`
- [ ] Public domain generated and tested
