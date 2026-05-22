FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/common.txt requirements/gateway.txt /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/gateway.txt

COPY libs /app/libs
COPY demo /app/demo
COPY services/gateway /app/services/gateway

ENV PYTHONPATH=/app/libs:/app/services/gateway

WORKDIR /app/services/gateway
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 8443

ENTRYPOINT ["./docker-entrypoint.sh"]
