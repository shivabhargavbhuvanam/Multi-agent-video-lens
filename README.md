# Multi-Agent Video Lens

An AI-powered system that ingests YouTube videos and lets you query them using natural language. It extracts transcripts, visual scene descriptions, and object detections — then uses a multi-agent RAG pipeline to answer questions with precise timestamps.

---

## How It Works

```
YouTube Video
      │
      ▼
┌─────────────────────────────────────────────┐
│              hackathon.py                   │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Whisper  │  │ GPT-4V   │  │  YOLO    │  │
│  │Transcript│  │ Captions │  │ Objects  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └─────────────┴─────────────┘         │
│                     │                       │
│              Pinecone Vector DB              │
└─────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│              workflow.py                    │
│                                             │
│         GPT-4 Orchestrator                  │
│        ↙        ↓        ↘                  │
│  Transcript  Caption    YOLO                │
│   Agent      Agent      Agent               │
│        ↘        ↓        ↙                  │
│         Answer + Timestamp                  │
└─────────────────────────────────────────────┘
```

**Phase 1 — Ingestion (`hackathon.py`):**
The video is split into 20-second chunks. Each chunk is processed in parallel by three agents: Whisper transcribes the audio, GPT-4 Vision generates scene captions from keyframes, and YOLO detects objects. All outputs are embedded and stored in Pinecone with timestamp metadata.

**Phase 2 — Query (`workflow.py`):**
A LlamaIndex multi-agent workflow takes your question, routes it to the right agent (transcripts, captions, or YOLO), retrieves relevant chunks from Pinecone, and synthesizes an answer with the exact timestamp where the event occurs.

---

## Project Structure

```
├── hackathon.py          # Video ingestion pipeline
├── workflow.py           # Multi-agent Q&A workflow
├── yolo.py               # YOLO object detection utilities
├── llm.py                # LLM helpers
├── app.py                # App entry point
├── Embedding/
│   └── embedding_generator.py
├── templates/
│   └── index.html
├── test/                 # API connection tests
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg` on macOS)

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_huggingface_token
```

---

## Usage

### Step 1 — Ingest a video

```bash
python hackathon.py
```

This downloads the video, processes it into chunks, runs all three agents, and pushes embeddings to Pinecone. Only needs to run once per video.

### Step 2 — Query the video

```bash
python workflow.py
```

You'll be prompted to ask questions:

```
Hello! I can help you query the video. Ask me anything about:
  • What happens visually in scenes  (image captions)
  • What is being said               (transcripts)
  • What objects appear on screen    (YOLO detection)

> What is the character saying at the beginning?
```

Type `exit` to quit.

---

## Agents

| Agent | Triggered by | Data source |
|---|---|---|
| **Transcript** | Questions about speech, dialogue, what was said | Whisper audio transcription |
| **Image Captioning** | Questions about scenes, appearance, actions, colours | GPT-4 Vision keyframe captions |
| **YOLO** | Explicit questions about detected objects | YOLOv11 object detection |

---

## Tech Stack

- **LlamaIndex** — workflow orchestration and RAG query engine
- **Pinecone** — vector storage and retrieval
- **OpenAI GPT-4** — orchestration, routing, and answer synthesis
- **OpenAI Whisper** (via HuggingFace) — audio transcription
- **YOLOv11** (Ultralytics) — real-time object detection
- **HuggingFace BGE** — text embeddings (`BAAI/bge-base-en-v1.5`)
- **yt-dlp** — video download
- **MoviePy / OpenCV** — video processing and keyframe extraction
