import os
import sys

LIBS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs"))
sys.path.insert(0, LIBS)

from reclamation_common.pre_process import pre_process


def test_pre_process_masks_phone():
    text = "Appelez-moi au 0555123456 pour la panne"
    out = pre_process(text)
    assert "<PHONE>" in out


def test_pre_process_empty():
    assert pre_process(None) == ""
    assert pre_process(123) == ""
