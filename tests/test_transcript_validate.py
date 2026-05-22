import sys
from pathlib import Path

LIBS = Path(__file__).resolve().parents[1] / "libs"
sys.path.insert(0, str(LIBS))

from reclamation_common.transcript_validate import validate_transcript


def test_valid_darija_phrase():
    ok, reason = validate_transcript("عندي مشكل فالانترنت من شهر", language="darija")
    assert ok is True
    assert reason is None


def test_reject_hallucination():
    ok, reason = validate_transcript(
        "bonjour, je souhaite avoir des informations sur les offres internet",
        language="fr",
    )
    assert ok is False
    assert reason == "hallucination"


def test_reject_too_short():
    ok, reason = validate_transcript("ok", language="darija")
    assert ok is False
    assert reason in ("too_short", "too_few_words")
