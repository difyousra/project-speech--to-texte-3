import base64
import io
import os
import sys
from functools import lru_cache

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

LIBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from app.config import API_KEY, HOST, PORT, SAMPLE_RATE  # noqa: E402
from app.vad_engine import create_vad_engine, load_audio_from_bytes  # noqa: E402


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class SegmentResponse(BaseModel):
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    speech_ratio: float
    avg_energy: float
    audio_base64: str


class VadResponse(BaseModel):
    segments: list[SegmentResponse]
    sample_rate: int


@lru_cache(maxsize=1)
def get_engine():
    return create_vad_engine()


app = FastAPI(title="VAD Service", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "vad"}


@app.post("/api/v1/segment", response_model=VadResponse)
async def segment_audio(
    file: UploadFile = File(...),
    _: None = Depends(verify_api_key),
):
    data = await file.read()
    audio = load_audio_from_bytes(data)
    engine = get_engine()
    segments = engine.segment_file(audio)

    out: list[SegmentResponse] = []
    for idx, seg in enumerate(segments):
        buf = io.BytesIO()
        sf.write(buf, seg.audio, SAMPLE_RATE, format="WAV")
        out.append(
            SegmentResponse(
                index=idx,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                duration_ms=seg.end_ms - seg.start_ms,
                speech_ratio=seg.speech_ratio,
                avg_energy=seg.avg_energy,
                audio_base64=base64.b64encode(buf.getvalue()).decode("ascii"),
            )
        )
    return VadResponse(segments=out, sample_rate=SAMPLE_RATE)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
