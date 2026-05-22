import sys
from pathlib import Path

LIBS = Path(__file__).resolve().parents[1] / "libs"
sys.path.insert(0, str(LIBS))

from reclamation_common.stt_metrics import char_error_rate, word_error_rate


def test_wer_identical():
    assert word_error_rate("modem internet", "modem internet") == 0.0


def test_wer_one_substitution():
    wer = word_error_rate("راني مخلص", "راني خلص")
    assert 0 < wer < 1
