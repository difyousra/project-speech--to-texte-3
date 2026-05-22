"""
Convention d'écriture du Whisper Darija fine-tuné (Algérie Télécom).

- Darija / arabe : graphie arabe ou latine selon l'oral.
- Emprunts français (internet, modem, facture, …) : toujours en français (latin),
  même si l'oral est en arabe (ex. « للانترنت » → « internet »).
"""

from __future__ import annotations

import re

# (motif arabe/latin alternatif, forme française cible)
_FRENCH_LOANWORD_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # Internet (oral arabe → graphie FR du fine-tuning)
    (re.compile(r"للانترنت|الانترنت|إنترنت|انترنت", re.IGNORECASE), "internet"),
    # Équipement & offres (emprunts FR fréquents en réclamation AT)
    (re.compile(r"مودم|المودم|موديم", re.IGNORECASE), "modem"),
    (re.compile(r"فاتورة|الفاتورة", re.IGNORECASE), "facture"),
    (re.compile(r"تيليفون|تليفون|تلفون", re.IGNORECASE), "téléphone"),
    (re.compile(r"راوتر|روتر", re.IGNORECASE), "routeur"),
    (re.compile(r"ويفي|واي\s*فاي", re.IGNORECASE), "wifi"),
    (re.compile(r"انستالاسيون", re.IGNORECASE), "installation"),
    (re.compile(r"فيبي|فibre", re.IGNORECASE), "fibre"),
    (re.compile(r"ادسال", re.IGNORECASE), "ADSL"),
]


def apply_stt_convention(text: str) -> str:
    """Applique la convention fine-tuning : termes FR en graphie française."""
    if not text or not text.strip():
        return text
    out = text
    for pattern, replacement in _FRENCH_LOANWORD_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    return re.sub(r"\s+", " ", out).strip()
