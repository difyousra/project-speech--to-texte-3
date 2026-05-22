import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

LIBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from app.config import API_KEY, HOST, PORT, STT_DEFAULT_LANGUAGE, STT_TORCH_THREADS  # noqa: E402
from app.transcribe import load_audio_bytes, transcribe_audio  # noqa: E402


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class TranscribeRequest(BaseModel):
    language_hint: Optional[str] = None
    partial: bool = False
    session_id: Optional[str] = None


_model_ready = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model_ready
    _model_ready = False
    torch.set_num_threads(STT_TORCH_THREADS)
    preload = "darija" if STT_DEFAULT_LANGUAGE in ("darija", "dz") else STT_DEFAULT_LANGUAGE
    if preload not in ("fr", "ar", "darija"):
        preload = "darija"
    logger.info("Préchargement Whisper %s (CPU)…", preload)
    try:
        from app.transcribe import get_pool

        pool = get_pool()
        pool._load_hf(preload)  # type: ignore[arg-type]
        _model_ready = True
        logger.info("Modèle Whisper %s prêt", preload)
    except Exception as exc:
        logger.warning("Préchargement STT échoué: %s", exc)
    yield
    _model_ready = False


app = FastAPI(title="STT Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if _model_ready else "loading",
        "service": "stt",
        "ready": _model_ready,
        "default_language": STT_DEFAULT_LANGUAGE,
    }


@app.get("/ready")
def ready():
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Modèle Whisper en chargement (≈30–60 s après démarrage)",
        )
    return {"ready": True}


@app.post("/api/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_hint: Optional[str] = None,
    partial: bool = False,
    session_id: Optional[str] = None,
    _: None = Depends(verify_api_key),
):
    data = await file.read()
    audio = load_audio_bytes(data)
    return transcribe_audio(
        audio,
        language_hint=language_hint,
        partial=partial,
        session_id=session_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
