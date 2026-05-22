import sys
from pathlib import Path

LIBS = Path(__file__).resolve().parents[1] / "libs"
sys.path.insert(0, str(LIBS))

from reclamation_common.wait_time import extract_waiting_time, waiting_days_estimate


def test_chhr_mli_implicit_month():
    text = "ياودي عندي شهر ملي ركبولي modem وراه نمخلص"
    val, unit = extract_waiting_time(text)
    assert val == 1
    assert unit == "mois"
    assert waiting_days_estimate(val, unit) == 30
