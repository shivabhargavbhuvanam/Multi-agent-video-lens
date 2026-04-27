FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv is a Rust-based installer that bypasses pip's resolver entirely
RUN pip install uv

WORKDIR /app
COPY requirements.txt .
# --no-deps skips the resolver; all transitive deps are already in the freeze file
RUN uv pip install --system --no-deps -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "--workers", "1", "app:app"]
