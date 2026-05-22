FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/common.txt requirements/nlp.docker.txt /app/requirements/
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements/nlp.docker.txt

COPY libs /app/libs
COPY services/nlp /app/services/nlp

ENV PYTHONPATH=/app/libs:/app/services/nlp
ENV PROJECT_ROOT=/app
ENV DEVICE=cpu
ENV CUDA_VISIBLE_DEVICES=

WORKDIR /app/services/nlp
EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
