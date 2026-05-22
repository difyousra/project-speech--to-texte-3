import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app")).resolve()

WHISPER_FR_PATH = os.getenv("WHISPER_FR_PATH", str(PROJECT_ROOT / "whisper-finetuned/FR/checkpoint-6250"))
WHISPER_AR_PATH = os.getenv("WHISPER_AR_PATH", str(PROJECT_ROOT / "whisper-finetuned/AR/checkpoint-2728"))
WHISPER_DZ_PATH = os.getenv("WHISPER_DZ_PATH", str(PROJECT_ROOT / "whisper-finetuned/Darja/checkpoint-1100"))

STT_USE_FASTER_WHISPER = os.getenv("STT_USE_FASTER_WHISPER", "false").lower() in ("1", "true", "yes")
STT_QUANTIZE_DYNAMIC = os.getenv("STT_QUANTIZE_DYNAMIC", "true").lower() in ("1", "true", "yes")
STT_DEFAULT_LANGUAGE = os.getenv("STT_DEFAULT_LANGUAGE", "darija").lower()
STT_MAX_AUDIO_SEC = float(os.getenv("STT_MAX_AUDIO_SEC", "12"))
STT_MAX_NEW_TOKENS_PARTIAL = int(os.getenv("STT_MAX_NEW_TOKENS_PARTIAL", "48"))
STT_MAX_NEW_TOKENS_FULL = int(os.getenv("STT_MAX_NEW_TOKENS_FULL", "128"))
STT_TORCH_THREADS = int(os.getenv("STT_TORCH_THREADS", "4"))
MAX_LOADED_MODELS = int(os.getenv("STT_MAX_LOADED_MODELS", "1"))
DEVICE = os.getenv("DEVICE", "cuda")
API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("STT_HOST", "0.0.0.0")
PORT = int(os.getenv("STT_PORT", "8002"))

LANG_MAP = {
    "fr": WHISPER_FR_PATH,
    "ar": WHISPER_AR_PATH,
    "darija": WHISPER_DZ_PATH,
    "dz": WHISPER_DZ_PATH,
}

SAMPLE_RATE = 16000
