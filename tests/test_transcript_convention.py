import sys
from pathlib import Path

LIBS = Path(__file__).resolve().parents[1] / "libs"
sys.path.insert(0, str(LIBS))

from reclamation_common.transcript_convention import apply_stt_convention


def test_internet_arabic_to_french():
    assert apply_stt_convention("مازال مطلقتوليش للانترنت") == "مازال مطلقتوليش internet"


def test_modem_kept_latin():
    assert "modem" in apply_stt_convention("ركبولي modem و راني مخلص")


def test_already_french_unchanged():
    t = "ياودي عندي شهر ملي ركبولي modem و مازال مطلقتوليش internet"
    assert apply_stt_convention(t) == t
