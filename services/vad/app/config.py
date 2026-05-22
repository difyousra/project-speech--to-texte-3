import os

VAD_ENGINE = os.getenv("VAD_ENGINE", "silero")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHUNK_MS = int(os.getenv("CHUNK_MS", "30"))
ENERGY_THRESHOLD = float(os.getenv("ENERGY_THRESHOLD", "0.006"))
MIN_AUDIO_MS = int(os.getenv("MIN_AUDIO_MS", "1500"))
MAX_AUDIO_MS = int(os.getenv("MAX_AUDIO_MS", "9000"))
SILENCE_LIMIT_MS = int(os.getenv("SILENCE_LIMIT_MS", "300"))
API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("VAD_HOST", "0.0.0.0")
PORT = int(os.getenv("VAD_PORT", "8001"))
