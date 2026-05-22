"""Métriques STT (WER / CER) avec normalisation adaptée au darija / code-switching."""

from __future__ import annotations

import re
import unicodedata


def normalize_for_metrics(text: str) -> str:
    """Normalise pour comparaison WER/CER (proche de l'éval entraînement Whisper)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\u0600-\u06FF\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_metrics(reference).split()
    hyp = normalize_for_metrics(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(ref, hyp) / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_metrics(reference).replace(" ", "")
    hyp = normalize_for_metrics(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(list(ref), list(hyp)) / len(ref)


def try_jiwer_metrics(reference: str, hypothesis: str) -> tuple[float, float] | None:
    try:
        import jiwer

        ref = normalize_for_metrics(reference)
        hyp = normalize_for_metrics(hypothesis)
        return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)
    except ImportError:
        return None
