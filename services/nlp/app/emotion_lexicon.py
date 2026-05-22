import re
from typing import Literal

FRUSTRATION_PATTERNS = [
    r"\bfrustr",
    r"\bénerv",
    r"\binsupportable\b",
    r"\bmarre\b",
    r"\burgent\b",
    r"\bcoupure\b",
    r"\bpanne\b",
    r"\bça fait\b",
    r"\bdepuis\b",
]

ANGER_PATTERNS = [
    r"\bcolère\b",
    r"\bfurieux\b",
    r"\bscandale\b",
    r"\binacceptable\b",
    r"\bhonte\b",
    r"\barnaque\b",
]

SATISFACTION_PATTERNS = [
    r"\bmerci\b",
    r"\bbravo\b",
    r"\bexcellent\b",
    r"\bsatisf",
    r"\bcontent\b",
    r"\bشكر",
    r"\bممتاز",
    r"\bبارك",
]

DARJA_FRUSTRATION = [
    r"\brah y3ani\b",
    r"\bma3adch\b",
    r"\bmakach\b",
    r"\bmazalna\b",
    r"\bwa9ef\b",
    r"\bma kayen\b",
    r"\b3andi\b",
    r"\b3ndna\b",
]

DARJA_ANGER = [
    r"\b7ram\b",
    r"\b3ayb\b",
    r"\bma y7mlch\b",
]


def _match_any(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered, re.IGNORECASE) for p in patterns)


def infer_emotion(
    text: str,
    sentiment: Literal["positive", "negative", "neutral"],
) -> str:
    if sentiment == "positive":
        if _match_any(text, SATISFACTION_PATTERNS):
            return "satisfaction"
        return "satisfaction"

    if sentiment == "neutral":
        return "neutral"

    if _match_any(text, DARJA_ANGER + ANGER_PATTERNS):
        return "anger"
    if _match_any(text, DARJA_FRUSTRATION + FRUSTRATION_PATTERNS):
        return "frustration"
    return "frustration"
