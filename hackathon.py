from huggingface_hub import InferenceClient
import base64
import ssl
import logging
import time
from datetime import datetime
from llama_index.core.schema import TextNode
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from pinecone import ServerlessSpec
from pinecone import Pinecone
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import yt_dlp
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.models import resnet50, ResNet50_Weights
import torch
import cv2
from pytubefix.cli import on_progress
from pytubefix import YouTube
from pytube import YouTube
from moviepy import VideoFileClip
import requests
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()


# ── Logger Setup ───────────────────────────────────────────────────────────────

def setup_chunk_logger(log_path: str) -> logging.Logger:
    """Create a logger that writes to both the log file and console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("chunk_logger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # avoid duplicate handlers on re-runs

    # File handler — writes every log line to processing.log
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))

    # Console handler — mirrors to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class ChunkLogger:
    """
    Tracks start/end wall-clock timestamps for every processing step
    within one chunk and writes structured entries to the shared log file.

    Usage:
        clog = ChunkLogger(logger, chunk_index=1, chunk_name="video_0_20")
        clog.begin("audio")
        ...do work...
        clog.end("audio")
        clog.complete()
    """

    def __init__(self, logger: logging.Logger, chunk_index: int, chunk_name: str):
        self.logger = logger
        self.label = f"[CHUNK-{chunk_index}]"
        self.chunk_name = chunk_name
        self._step_start: float = 0.0
        self._chunk_start: float = time.perf_counter()

        self.logger.info(
            f"\n{'='*60}\n"
            f"{self.label} chunk={chunk_name}  "
            f"started_at={datetime.now().strftime('%H:%M:%S.%f')[:-3]}\n"
            f"{'='*60}"
        )

    def begin(self, step: str) -> str:
        """Record start of a step. Returns the timestamp string."""
        self._step_start = time.perf_counter()
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logger.info(f"{self.label} {step:<12} START={ts}")
        return ts

    def end(self, step: str) -> str:
        """Record end of a step with elapsed duration. Returns the timestamp string."""
        elapsed = time.perf_counter() - self._step_start
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logger.info(
            f"{self.label} {step:<12} END  ={ts}  duration={elapsed:.3f}s"
        )
        return ts

    def complete(self):
        """Log the total wall-clock time for the entire chunk."""
        total = time.perf_counter() - self._chunk_start
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logger.info(
            f"{self.label} COMPLETE      total_wall_time={total:.3f}s  "
            f"finished_at={ts}\n"
        )


# ── Global setup ───────────────────────────────────────────────────────────────

model = YOLO("yolo11n.pt")
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
ssl._create_default_https_context = ssl._create_stdlib_context
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Shared logger instance — set when VideoProcessingWorkflow is constructed
_chunk_logger: logging.Logger = None


# ── Pinecone / Embeddings ──────────────────────────────────────────────────────

def create_embeddings(data):
    # huggingface model for embedding
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5")

    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)

    existing_indexes = [i.name for i in pc.list_indexes()]
    if "video-analysis-index-v2" not in existing_indexes:
        pc.create_index(
            "video-analysis-index-v2",
            dimension=768,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    pinecone_index = pc.Index("video-analysis-index-v2")

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace="Default"
    )

    nodes = []
    for d in data:
        nodes.append(TextNode(
            text="Timestamp: " +
            str(d['timestamp']) + "\n" + 'video_id: ' +
            str(d['video_id']) + "\n\n" + d['text'],
            metadata={
                "timestamp": d['timestamp'],
                "video_id": d['video_id'],
                "agent": d['agent']
            }
        ))

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)


def generate_embedding(agent, video_id, base_path):
    agent_folder_mapping = {
        'image_captioning': 'image_captionings',
        'transcripts': 'transcripts',
        'yolo': 'yolo_outputs'
    }

    if agent not in agent_folder_mapping:
        raise ValueError(f"Unknown agent '{agent}'")

    path = os.path.join(base_path, 'chunks', agent_folder_mapping[agent])
    os.makedirs(path, exist_ok=True)
    data_array = []

    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            timestamp = parts[1]
            with open(os.path.join(path, filename), 'r') as file:
                text = file.read()

            entry = {
                'text': f"Timestamp: {timestamp}\nvideo_id: {video_id}\n\n{text.strip()}",
                'timestamp': timestamp,
                'video_id': video_id,
                'agent': agent
            }
            data_array.append(entry)

    create_embeddings(data_array)


# ── Video Processing Workflow ──────────────────────────────────────────────────

class VideoProcessingWorkflow:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        self.setup_directories()

        # Initialise the shared chunk logger
        log_path = os.path.join(output_dir, "processing.log")
        global _chunk_logger
        _chunk_logger = setup_chunk_logger(log_path)
        _chunk_logger.info(
            f"\n{'#'*60}\n"
            f"# NEW RUN — {datetime.now().isoformat()}\n"
            f"# video  = {video_path}\n"
            f"# output = {output_dir}\n"
            f"{'#'*60}"
        )

        # 1-based chunk counter (safe in single-threaded asyncio)
        self._chunk_index = 0

    def setup_directories(self):
        directories = [self.output_dir,
                       os.path.join(self.output_dir, "videos"),
                       os.path.join(self.output_dir, "audios"),
                       os.path.join(self.output_dir, "transcripts"),
                       os.path.join(self.output_dir, "images"),
                       os.path.join(self.output_dir, "image_captionings"),
                       os.path.join(self.output_dir, "yolo_outputs")]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    async def process_subclip(self, start_time, end_time, subclip_path, audio_path):
        self._chunk_index += 1
        chunk_name = f"video_{start_time}_{end_time}"
        clog = ChunkLogger(_chunk_logger, self._chunk_index, chunk_name)

        try:
            with VideoFileClip(self.video_path).subclip(start_time, end_time) as clip:

                # ── VIDEO chunk ───────────────────────────────────────────────
                clog.begin("video")
                clip.write_videofile(subclip_path, codec='mpeg4')
                clog.end("video")

                # ── AUDIO chunk ───────────────────────────────────────────────
                clog.begin("audio")
                clip.audio.write_audiofile(audio_path)
                clog.end("audio")

                # ── TRANSCRIPT ────────────────────────────────────────────────
                clog.begin("transcript")
                transcript = query(audio_path)
                transcript_path = os.path.join(
                    self.output_dir, "transcripts", f"transcript_{start_time}_{end_time}.txt")
                with open(transcript_path, 'w') as f:
                    f.write(transcript['text'])
                clog.end("transcript")

                print(f"Processed and transcribed video and audio from {start_time} to {end_time} seconds.")

                # ── KEYFRAMES ─────────────────────────────────────────────────
                images_dir = os.path.join(
                    self.output_dir, "images", f"video_{start_time}_{end_time}")
                os.makedirs(images_dir, exist_ok=True)

                clog.begin("keyframe")
                image_captioning_content = self.extract_keyframes_dl(
                    subclip_path, images_dir)
                clog.end("keyframe")

                print("Final caption:")
                print(image_captioning_content)

                # ── IMAGE CAPTION ─────────────────────────────────────────────
                clog.begin("caption")
                image_captioning_path = os.path.join(
                    self.output_dir, "image_captionings", f"imagecaptioning_{start_time}_{end_time}.txt")
                os.makedirs(os.path.dirname(image_captioning_path), exist_ok=True)
                with open(image_captioning_path, 'w') as f:
                    f.write(image_captioning_content)
                clog.end("caption")

                print(f"Saved image captioning for {start_time} to {end_time} seconds.")

        except Exception as e:
            _chunk_logger.error(f"{clog.label} ERROR — {str(e)}")
            print(f"Error processing media from {start_time} to {end_time} seconds: {str(e)}")
        finally:
            clog.complete()

    async def split_video_and_audio(self):
        try:
            with VideoFileClip(self.video_path) as clip:
                duration = int(clip.duration)
                tasks = []
                for start_time in range(0, duration, 20):
                    end_time = min(start_time + 20, duration)
                    subclip_path = os.path.join(
                        self.output_dir, "videos", f"video_{start_time}_{end_time}.mp4")
                    audio_path = os.path.join(
                        self.output_dir, "audios", f"audio_{start_time}_{end_time}.mp3")
                    task = asyncio.create_task(self.process_subclip(
                        start_time, end_time, subclip_path, audio_path))
                    tasks.append(task)
                await asyncio.gather(*tasks)
        except Exception as e:
            if _chunk_logger:
                _chunk_logger.error(f"[SPLITTER] Failed to open video: {str(e)}")
            print(f"Error opening video file: {str(e)}")

    def extract_keyframes_dl(self, video_path, output_dir, threshold=0.5):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
        print(f"Cleared all contents in {output_dir}")

        print("Generating keyframes...")
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()

        if not ret:
            print("Failed to read video")
            cap.release()
            return ""

        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet.eval()

        frame_count = 0
        prev_features = self.get_frame_features(prev_frame, resnet, transform)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            features = self.get_frame_features(frame, resnet, transform)
            similarity = torch.nn.functional.cosine_similarity(
                prev_features, features, dim=1)

            if similarity.item() < threshold:
                frame_path = os.path.join(
                    output_dir, f"keyframe_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"Saved keyframe {frame_count} at {frame_path}")
                prev_features = features

            frame_count += 1

        cap.release()
        folder_path = output_dir
        image_paths = os.listdir(folder_path)
        print(image_paths)

        base64_images = [encode_image(os.path.join(
            folder_path, image_path)) for image_path in image_paths]
        print(len(base64_images))

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These are the key frames images which indicate change of scenes when extracted from the set of image frames. Give me an good overall description summary of what is happening in the video so that whenever I search for it in the future, i can know what is happening here."
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                    for base64_image in base64_images
                ]
            }],
            "max_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_dict = response.json()

        if ("choices" in response_dict):
            content = response_dict['choices'][0]['message']['content']
            return content
        else:
            print(response_dict)
            raise Exception("Error in OpenAI response")

    def get_frame_features(self, frame, model, transform):
        """Extract features from a frame using the specified model and transformation."""
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            return model(frame_tensor)


def download_video(url, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(output_path, 'movie.mp4'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ.get("HF_TOKEN", ""),
)


def query(filename):
    output = hf_client.automatic_speech_recognition(
        filename, model="openai/whisper-large-v3"
    )
    return {"text": output.text}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_yolo_chunks(video_id='bahubali', source='sources/bahubali/'):
    path = os.path.join(source, 'chunks', 'images')
    model = YOLO("yolo11n.pt")
    data = []
    for filename in os.listdir(path):
        print(filename)
        timestamp = filename.split('_')[1]
        timestamp_end = filename.split('_')[2]

        filepath = path+'/'+filename
        entry = {
            'text': '',
            'timestamp': timestamp,
            'video_id': video_id,
            'agent': 'yolo'
        }
        text = 'The yolo objects in the frames from timestamp'+timestamp+'to '+timestamp_end
        for file in os.listdir(filepath):
            try:
                results = model(filepath+'/'+file)
                text += str(results[0].to_json())
            except Exception as e:
                print(f"Skipping {file}: {e}")
        entry['text'] = text
        data.append(entry)
    create_embeddings(data)


if __name__ == "__main__":
    video_id = 'bahubali'
    source = f"sources/{video_id}/"
    video_path = source + "movie.mp4"
    output_dir = source + "chunks"

    print("YOUTUBE: Download started")
    download_video(
        'https://www.youtube.com/watch?v=aQQUl22NNGE', source)

    workflow = VideoProcessingWorkflow(video_path, output_dir)
    asyncio.run(workflow.split_video_and_audio())
    create_yolo_chunks()  # this also pushes embeddings
    generate_embedding('image_captioning', video_id, source)
    generate_embedding('transcripts', video_id, source)