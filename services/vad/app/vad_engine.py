import io
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config import (
    CHUNK_MS,
    ENERGY_THRESHOLD,
    MAX_AUDIO_MS,
    MIN_AUDIO_MS,
    SAMPLE_RATE,
    SILENCE_LIMIT_MS,
    VAD_ENGINE,
)


class VadEngineType(str, Enum):
    SILERO = "silero"
    RMS = "rms"


@dataclass
class SpeechSegment:
    audio: np.ndarray
    start_ms: int
    end_ms: int
    speech_ratio: float
    avg_energy: float


def rms_energy(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


class SileroVAD:
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        import torch

        self.sample_rate = sample_rate
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.get_speech_timestamps = utils[0]

    def segment_file(self, audio: np.ndarray) -> list[SpeechSegment]:
        import torch

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        tensor = torch.from_numpy(audio)
        timestamps = self.get_speech_timestamps(
            tensor,
            self.model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=MIN_AUDIO_MS,
            max_speech_duration_s=MAX_AUDIO_MS / 1000.0,
            min_silence_duration_ms=SILENCE_LIMIT_MS,
        )
        segments: list[SpeechSegment] = []
        for ts in timestamps:
            start = int(ts["start"])
            end = int(ts["end"])
            chunk = audio[start:end]
            dur_ms = int((end - start) / self.sample_rate * 1000)
            if dur_ms < MIN_AUDIO_MS:
                continue
            energy = rms_energy(chunk)
            segments.append(
                SpeechSegment(
                    audio=chunk,
                    start_ms=int(start / self.sample_rate * 1000),
                    end_ms=int(end / self.sample_rate * 1000),
                    speech_ratio=0.9,
                    avg_energy=energy,
                )
            )
        return segments


class RmsVAD:
    """Port de la logique test_vad_chunks.py pour traitement offline."""

    def __init__(self):
        self.block_size = int(SAMPLE_RATE * CHUNK_MS / 1000)

    def segment_file(self, audio: np.ndarray) -> list[SpeechSegment]:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        blocks = [
            audio[i : i + self.block_size]
            for i in range(0, len(audio) - self.block_size, self.block_size)
        ]
        if not blocks:
            return []

        segments: list[SpeechSegment] = []
        recording = False
        current: list[np.ndarray] = []
        flags: list[bool] = []
        silence_ms = 0
        segment_ms = 0

        for block in blocks:
            energy = rms_energy(block)
            is_speech = energy > ENERGY_THRESHOLD

            if is_speech and not recording:
                recording = True
                current = [block]
                flags = [is_speech]
                silence_ms = 0
                segment_ms = CHUNK_MS
                continue

            if recording:
                current.append(block)
                flags.append(is_speech)
                segment_ms += CHUNK_MS
                silence_ms = 0 if is_speech else silence_ms + CHUNK_MS

                cut_silence = segment_ms >= MIN_AUDIO_MS and silence_ms >= SILENCE_LIMIT_MS
                cut_max = segment_ms >= MAX_AUDIO_MS
                if cut_silence or cut_max:
                    seg = self._make_segment(current, flags)
                    if seg:
                        segments.append(seg)
                    recording = False
                    current = []
                    flags = []
                    silence_ms = 0
                    segment_ms = 0

        if recording and current:
            seg = self._make_segment(current, flags)
            if seg:
                segments.append(seg)
        return segments

    def _make_segment(self, blocks: list[np.ndarray], flags: list[bool]) -> SpeechSegment | None:
        audio = np.concatenate(blocks, axis=0)
        dur_ms = int(len(audio) / SAMPLE_RATE * 1000)
        avg_energy = rms_energy(audio)
        speech_ratio = sum(flags) / max(len(flags), 1)
        if dur_ms < MIN_AUDIO_MS or speech_ratio < 0.2:
            return None
        return SpeechSegment(
            audio=audio.astype(np.float32),
            start_ms=0,
            end_ms=dur_ms,
            speech_ratio=speech_ratio,
            avg_energy=avg_energy,
        )


def create_vad_engine() -> SileroVAD | RmsVAD:
    if VAD_ENGINE == VadEngineType.SILERO.value:
        try:
            return SileroVAD()
        except Exception:
            return RmsVAD()
    return RmsVAD()


def load_audio_from_bytes(data: bytes) -> np.ndarray:
    import os
    import sys

    libs = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
    if libs not in sys.path:
        sys.path.insert(0, libs)
    from reclamation_common.audio_io import load_audio_from_bytes as _load

    return _load(data, SAMPLE_RATE)
