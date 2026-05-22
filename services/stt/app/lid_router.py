from collections import deque
from typing import Literal

import numpy as np

LanguageCode = Literal["fr", "ar", "darija"]

LID_WINDOW_VOTES = 3
LID_CONFIDENCE_THRESHOLD = 0.5


class LanguageRouter:
    """Routage audio avec hystérésis sur votes consécutifs."""

    def __init__(self, window_size: int = LID_WINDOW_VOTES, default_language: LanguageCode = "darija"):
        self._votes: deque[LanguageCode] = deque(maxlen=window_size)
        self._current: LanguageCode = default_language if default_language in ("fr", "ar", "darija") else "darija"

    @property
    def current_language(self) -> LanguageCode:
        return self._current

    def _heuristic_lid(self, audio: np.ndarray, text_hint: str = "") -> tuple[LanguageCode, float]:
        if text_hint.strip():
            lowered = text_hint.lower()
            darija_markers = ["rah", "3ndi", "makench", "wga3d", "te3", "b l'", "ghir", "bnadem"]
            if any(m in lowered for m in darija_markers):
                return "darija", 0.7
            arabic_chars = sum(1 for c in text_hint if "\u0600" <= c <= "\u06FF")
            latin_chars = sum(1 for c in text_hint if c.isascii() and c.isalpha())
            if arabic_chars > latin_chars * 2:
                return "ar", 0.65
            if latin_chars > arabic_chars:
                return "fr", 0.65

        if len(audio) < 1600:
            return self._current, 0.4

        energy = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if energy < 1e-5:
            return self._current, 0.3

        return self._current, 0.45

    def update(self, detected: LanguageCode, confidence: float) -> LanguageCode:
        if confidence >= LID_CONFIDENCE_THRESHOLD:
            self._votes.append(detected)
        if len(self._votes) == self._votes.maxlen:
            counts: dict[LanguageCode, int] = {}
            for lang in self._votes:
                counts[lang] = counts.get(lang, 0) + 1
            winner = max(counts, key=counts.get)
            if counts[winner] >= 2:
                self._current = winner
        return self._current

    def route(self, audio: np.ndarray, whisper_detected: str | None = None, text_hint: str = "") -> LanguageCode:
        whisper_map = {"french": "fr", "fr": "fr", "arabic": "ar", "ar": "ar"}
        if whisper_detected:
            lang = whisper_map.get(whisper_detected.lower(), "darija" if whisper_detected == "ar" else whisper_detected)
            if lang in ("fr", "ar"):
                return self.update(lang, 0.8)  # type: ignore[arg-type]

        heuristic_lang, conf = self._heuristic_lid(audio, text_hint)
        return self.update(heuristic_lang, conf)
