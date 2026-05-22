import os
import sys

SERVICES_NLP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "services", "nlp"))
sys.path.insert(0, SERVICES_NLP)

from app.emotion_lexicon import infer_emotion


def test_frustration_darija():
    assert infer_emotion("rah y3ani b l'adsl makach", "negative") == "frustration"


def test_satisfaction_positive():
    assert infer_emotion("merci bcp service excellent", "positive") == "satisfaction"
