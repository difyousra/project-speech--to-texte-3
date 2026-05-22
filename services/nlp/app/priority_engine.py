from typing import Literal, Optional

from reclamation_common.label_maps import (
    CATEGORY_SLUG_MAP,
    COMPLAINT_LEVEL1_INDEX,
    HANDLING_SLA_DAYS,
    ID2LABEL_LEVEL1,
    ID2LABEL_LEVEL2,
    INTENT_SLUG_MAP,
    REQUEST_TYPE_MAP,
    SERVICE_LABEL_FR,
    SERVICE_MAP,
    THEME_MAP,
)
from reclamation_common.schemas import ClassificationResult
from reclamation_common.wait_time import extract_waiting_time, waiting_days_estimate

URGENCY_KEYWORDS = [
    "urgent",
    "coupure",
    "panne",
    "coupe",
    "sans internet",
    "mort",
    "mekhdemch",
    "ma kayen",
    "rah m",
    "انقطاع",
    "عاجل",
]

HIGH_PRIORITY_CATEGORIES = {"pannes_coupures", "retards_installation"}


def _slug_from_level1(level1_label: str) -> str:
    return INTENT_SLUG_MAP.get(level1_label, "unknown")


def _slug_from_level2(level2_label: Optional[str]) -> Optional[str]:
    if not level2_label:
        return None
    return CATEGORY_SLUG_MAP.get(level2_label)


def compute_priority(
    level1_idx: int,
    category_slug: Optional[str],
    sentiment: str,
    waiting_days: Optional[int],
    text: str,
) -> tuple[Literal["low", "medium", "high", "critical"], bool]:
    urgency = False
    lowered = text.lower()

    if any(kw in lowered for kw in URGENCY_KEYWORDS):
        urgency = True

    if waiting_days is not None and waiting_days >= 7:
        urgency = True

    if level1_idx == COMPLAINT_LEVEL1_INDEX:
        if category_slug in HIGH_PRIORITY_CATEGORIES:
            if waiting_days and waiting_days >= 14:
                return "critical", True
            if urgency or sentiment == "negative":
                return "high", urgency or True
            return "medium", urgency
        if urgency:
            return "high", True
        return "medium", urgency

    if level1_idx == 1:
        return "high", True

    return "low", urgency


def build_classification(
    level1_idx: int,
    level2_idx: Optional[int],
    sentiment: str,
    text: str,
    confidence: dict[str, float],
) -> ClassificationResult:
    level1_label = ID2LABEL_LEVEL1[level1_idx]
    level2_label = ID2LABEL_LEVEL2[level2_idx] if level2_idx is not None else None

    intent = _slug_from_level1(level1_label)
    category_slug = _slug_from_level2(level2_label) if level1_idx == COMPLAINT_LEVEL1_INDEX else None

    wait_val, wait_unit = extract_waiting_time(text)
    waiting_days = waiting_days_estimate(wait_val, wait_unit)

    priority, urgency = compute_priority(
        level1_idx, category_slug, sentiment, waiting_days, text
    )

    service = SERVICE_MAP.get(category_slug) if category_slug else None
    service_label = SERVICE_LABEL_FR.get(service) if service else None
    if not service_label and level1_idx == COMPLAINT_LEVEL1_INDEX:
        service_label = "Service client Algérie Télécom"
    theme = THEME_MAP.get(category_slug) if category_slug else None
    request_type = REQUEST_TYPE_MAP.get(intent, "unknown")
    handling_sla_days = HANDLING_SLA_DAYS.get(priority, 5)

    return ClassificationResult(
        intent=intent,
        category=category_slug,
        service=service,
        service_label=service_label,
        priority=priority,
        urgency=urgency,
        theme=theme,
        request_type=request_type,
        level1_label=level1_label,
        level2_label=level2_label,
        confidence=confidence,
        waiting_days=waiting_days,
        waiting_unit=wait_unit,
        handling_sla_days=handling_sla_days,
    )
