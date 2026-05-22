import os

API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("GATEWAY_PORT", "8000"))

VAD_SERVICE_URL = os.getenv("VAD_SERVICE_URL", "http://localhost:8001")
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://localhost:8002")
NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", "http://localhost:8003")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
