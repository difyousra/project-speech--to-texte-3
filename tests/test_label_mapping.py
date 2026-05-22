import os
import sys

LIBS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "libs"))
SERVICES_NLP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "nlp"))
sys.path.insert(0, LIBS)
sys.path.insert(0, SERVICES_NLP)

from app.priority_engine import build_classification


def test_complaint_high_priority():
    result = build_classification(
        level1_idx=0,
        level2_idx=0,
        sentiment="negative",
        text="coupure internet urgente depuis 15 jours",
        confidence={"intent": 0.9, "category": 0.8},
    )
    assert result.intent == "reclamation"
    assert result.category == "pannes_coupures"
    assert result.urgency is True
    assert result.priority in ("high", "critical")


def test_positive_low_priority():
    result = build_classification(
        level1_idx=3,
        level2_idx=None,
        sentiment="positive",
        text="merci pour le service",
        confidence={"intent": 0.95},
    )
    assert result.intent == "positive"
    assert result.priority == "low"
