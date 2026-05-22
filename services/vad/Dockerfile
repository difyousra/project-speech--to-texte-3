FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/common.txt requirements/vad.docker.txt /app/requirements/
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements/vad.docker.txt

COPY libs /app/libs
COPY services/vad /app/services/vad

ENV PYTHONPATH=/app/libs:/app/services/vad
ENV VAD_ENGINE=rms
ENV DEVICE=cpu
ENV CUDA_VISIBLE_DEVICES=

WORKDIR /app/services/vad
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
