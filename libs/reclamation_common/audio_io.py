"""Chargement audio WAV/MP3/WebM via soundfile, sinon ffmpeg (sans torchcodec)."""

from __future__ import annotations

import io
import subprocess
import tempfile

import numpy as np

DEFAULT_SAMPLE_RATE = 16000


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32, copy=False)
    ratio = target_sr / sr
    new_len = int(len(audio) * ratio)
    if new_len < 1:
        return np.array([], dtype=np.float32)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _load_ffmpeg(data: bytes, sample_rate: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                tmp.name,
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-",
            ],
            capture_output=True,
            check=True,
        )
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio.copy()


def load_audio_from_bytes(data: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf

    buf = io.BytesIO(data)
    try:
        audio, sr = sf.read(buf, dtype="float32")
    except Exception:
        audio = _load_ffmpeg(data, sample_rate)
        sr = sample_rate

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _resample(audio, sr, sample_rate)
