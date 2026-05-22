import gc
from functools import lru_cache

from reclamation_common.pre_process import pre_process
from reclamation_common.schemas import AnalyzeResponse, ClassificationResult, SentimentResult

from app.config import (
    DEVICE,
    DZIRIBERT_PATH,
    LOW_MEMORY,
    LV1_MODEL_PATH,
    LV1_TOKENIZER_PATH,
    MULTITASK_TOKENIZER_PATH,
    MULTITASK_WEIGHTS_PATH,
    XLM_ROBERTA_BASE_PATH,
)
from app.emotion_lexicon import infer_emotion
from app.priority_engine import build_classification

_SENTIMENT_FROM_LEVEL1 = {
    0: "negative",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "neutral",
}


@lru_cache(maxsize=1)
def _get_classifier():
    if LOW_MEMORY:
        from app.models.lightweight_classifier import LightweightClassifier

        return LightweightClassifier(LV1_MODEL_PATH, LV1_TOKENIZER_PATH, DEVICE)

    from app.models.roberta_multitask import MultiTaskClassifier

    return MultiTaskClassifier(
        base_path=XLM_ROBERTA_BASE_PATH,
        weights_path=MULTITASK_WEIGHTS_PATH,
        tokenizer_path=MULTITASK_TOKENIZER_PATH,
        device=DEVICE,
    )


@lru_cache(maxsize=1)
def _get_sentiment_model():
    from app.models.dziribert import DziriBertSentiment

    return DziriBertSentiment(DZIRIBERT_PATH, DEVICE)


def _sentiment_from_level1(level1_idx: int, text: str) -> SentimentResult:
    label = _SENTIMENT_FROM_LEVEL1.get(level1_idx, "neutral")
    emotion = infer_emotion(text, label)
    return SentimentResult(sentiment=label, emotion=emotion, score=0.75)


def analyze_text(text: str, session_id: str | None = None) -> AnalyzeResponse:
    cleaned = pre_process(text)
    classifier = _get_classifier()
    p1_idx, p2_idx, confidence = classifier.predict(cleaned)

    if LOW_MEMORY:
        sentiment = _sentiment_from_level1(p1_idx, cleaned)
        classification = build_classification(
            p1_idx, p2_idx, sentiment.sentiment, cleaned, confidence
        )
    else:
        gc.collect()
        sentiment_model = _get_sentiment_model()
        sentiment_label, score = sentiment_model.predict(cleaned)
        emotion = infer_emotion(cleaned, sentiment_label)
        sentiment = SentimentResult(sentiment=sentiment_label, emotion=emotion, score=score)
        classification = build_classification(
            p1_idx, p2_idx, sentiment_label, cleaned, confidence
        )

    return AnalyzeResponse(text=cleaned, classification=classification, sentiment=sentiment)
