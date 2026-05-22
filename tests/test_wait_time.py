import os
import sys

LIBS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs"))
sys.path.insert(0, LIBS)

from reclamation_common.wait_time import extract_waiting_time, waiting_days_estimate


def test_french_days():
    val, unit = extract_waiting_time("ça fait 12 jours sans internet")
    assert val == 12
    assert waiting_days_estimate(val, unit) == 12
