FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/common.txt requirements/stt.docker.txt /app/requirements/
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements/stt.docker.txt

COPY libs /app/libs
COPY services/stt /app/services/stt

ENV PYTHONPATH=/app/libs:/app/services/stt
ENV PROJECT_ROOT=/app
ENV DEVICE=cpu
ENV CUDA_VISIBLE_DEVICES=
ENV STT_USE_FASTER_WHISPER=false

WORKDIR /app/services/stt
EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
