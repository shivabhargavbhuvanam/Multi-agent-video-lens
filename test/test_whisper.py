import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ.get("HF_TOKEN", ""),
)

output = client.automatic_speech_recognition("test_audio.mp3", model="openai/whisper-large-v3")
print(output)