"""Filtre les transcriptions vides, trop courtes ou hallucinations Whisper courantes."""

from __future__ import annotations

import re
import unicodedata

MIN_CHARS = 4
MIN_WORDS = 2
MIN_CONFIDENCE = 0.35

# Hallucinations fréquentes sur bruit / silence (FR/EN)
_HALLUCINATION_RE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"sous[- ]?titres",
        r"subtitles?\s+by",
        r"thank you for watching",
        r"merci d['']avoir regard",
        r"^bonjour,?\s+je souhaite avoir des informations",
        r"^\s*(\.\s*){3,}$",
        r"^[\W\d]+$",
    )
]

# Répétition excessive d'un même mot
_REPEAT_WORD_RE = re.compile(r"\b(\w{2,})\b(?:\s+\1\b){3,}", re.IGNORECASE)


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w\u0600-\u06FF]+", text, flags=re.UNICODE))


def validate_transcript(
    text: str,
    *,
    language: str = "darija",
    confidence: float = 1.0,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[bool, str | None]:
    """Retourne (validé, raison_rejet). Seules les transcriptions validées alimentent le NLP."""
    raw = (text or "").strip()
    if not raw:
        return False, "empty"

    normalized = unicodedata.normalize("NFKC", raw)
    if len(normalized) < MIN_CHARS:
        return False, "too_short"

    if _word_count(normalized) < MIN_WORDS:
        return False, "too_few_words"

    if confidence < min_confidence:
        return False, "low_confidence"

    for pat in _HALLUCINATION_RE:
        if pat.search(normalized):
            return False, "hallucination"

    if _REPEAT_WORD_RE.search(normalized):
        return False, "repetition"

    compact = re.sub(r"\s+", "", normalized)
    if len(compact) >= 8 and len(set(compact)) <= 2:
        return False, "repetition"

    # Ratio lettres utiles (arabe / latin / chiffres)
    useful = sum(
        1
        for c in normalized
        if c.isalnum() or "\u0600" <= c <= "\u06FF"
    )
    if useful < MIN_CHARS:
        return False, "no_speech_content"

    return True, None
