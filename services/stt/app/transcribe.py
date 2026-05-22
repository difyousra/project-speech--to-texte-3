import os
import sys
from functools import lru_cache

import numpy as np

LIBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from reclamation_common.transcript_convention import apply_stt_convention  # noqa: E402
from reclamation_common.transcript_validate import validate_transcript  # noqa: E402

from app.config import STT_DEFAULT_LANGUAGE, STT_MAX_AUDIO_SEC  # noqa: E402

SAMPLE_RATE = 16000
from app.lid_router import LanguageRouter  # noqa: E402

_router = LanguageRouter(
    default_language="darija" if STT_DEFAULT_LANGUAGE in ("darija", "dz") else STT_DEFAULT_LANGUAGE
)


@lru_cache(maxsize=1)
def get_pool():
    from app.whisper_pool import WhisperPool

    return WhisperPool()


def load_audio_bytes(data: bytes, sample_rate: int = 16000) -> np.ndarray:
    from reclamation_common.audio_io import load_audio_from_bytes as _load

    audio = _load(data, sample_rate)
    return trim_audio(audio, STT_MAX_AUDIO_SEC)


def trim_audio(audio: np.ndarray, max_sec: float) -> np.ndarray:
    max_samples = int(SAMPLE_RATE * max_sec)
    if len(audio) <= max_samples:
        return audio
    return audio[-max_samples:].copy()


def _resolve_language(language_hint: str | None, audio: np.ndarray) -> str:
    if language_hint and language_hint in ("fr", "ar", "darija", "dz"):
        return "darija" if language_hint == "dz" else language_hint
    if STT_DEFAULT_LANGUAGE in ("fr", "ar", "darija", "dz"):
        return "darija" if STT_DEFAULT_LANGUAGE == "dz" else STT_DEFAULT_LANGUAGE
    return _router.route(audio, None, text_hint=language_hint or "")


def transcribe_audio(
    audio: np.ndarray,
    language_hint: str | None = None,
    partial: bool = False,
    session_id: str | None = None,
) -> dict:
    audio = trim_audio(audio, STT_MAX_AUDIO_SEC)
    pool = get_pool()
    routed = _resolve_language(language_hint, audio)

    raw_text, confidence = pool.transcribe(audio, routed, partial=partial)  # type: ignore[arg-type]
    text = apply_stt_convention(raw_text)
    validated, reason = validate_transcript(text, language=routed, confidence=confidence)

    return {
        "text": text,
        "language": routed,
        "confidence": confidence,
        "is_partial": partial,
        "validated": validated,
        "validation_reason": reason,
        "session_id": session_id,
    }
