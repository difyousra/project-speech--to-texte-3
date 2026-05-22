import re
from typing import Optional, Tuple

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

PATTERNS_FR = [
    (r"(\d+)\s*(?:jour|jours)", "jour"),
    (r"(\d+)\s*(?:mois)", "mois"),
    (r"(\d+)\s*(?:an|ans)", "an"),
    (r"(\d+)\s*(?:heure|heures)", "heure"),
    (r"(\d+)\s*(?:semaine|semaines)", "semaine"),
    (r"j\+?\s*(\d+)", "j+"),
    (r"ça fait\s*(\d+)\s*(?:jour|jours)", "jour"),
]

PATTERNS_AR = [
    (r"(\d+)\s*(?:يوم|ايام)", "jour"),
    (r"(\d+)\s*(?:شهر|اشهر)", "mois"),
    (r"(\d+)\s*(?:عام|اعوام|سنوات)", "an"),
    (r"(\d+)\s*(?:ساعة|ساعات)", "heure"),
    (r"(\d+)\s*(?:اسبوع|اسابيع)", "semaine"),
    (r"منذ\s*(\d+)\s*(?:يوم|ايام|شهر|اشهر)", "depuis"),
]

PATTERNS_DARJA = [
    (r"(\d+)\s*(?:nhar|nhr|نهار)", "jour"),
    (r"(\d+)\s*(?:chhr|chhor|شهر)", "mois"),
    (r"(\d+)\s*(?:snin|3am|عام)", "an"),
    (r"(?:lberh|lebereh|barh|malbarh)", "jour"),
    (r"(?:3ndi|3ndna|rah)\s*(\d+)", "depuis"),
    # « عندي شهر » / « شهر ملي » sans chiffre explicite = 1 mois
    (r"(?:عندي\s+)?شهر(?:\s+ملي|\s+من|\s+لي)?\b", "mois_implicite"),
    (r"\bchhr\s+mli\b", "mois_implicite"),
]


def _normalize_digits(text: str) -> str:
    return text.translate(ARABIC_DIGITS)


def _extract_from_patterns(text: str, patterns: list[tuple[str, str]]) -> Tuple[Optional[int], Optional[str]]:
    for pattern, unit in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        if unit == "mois_implicite":
            return 1, "mois"
        if groups and groups[0].isdigit():
            return int(groups[0]), unit
        if unit == "jour" and not groups:
            return 1, unit
    return None, None


def extract_waiting_time(text: str) -> Tuple[Optional[int], Optional[str]]:
    if not text:
        return None, None
    normalized = _normalize_digits(text.lower())
    for patterns in (PATTERNS_FR, PATTERNS_AR, PATTERNS_DARJA):
        val, unit = _extract_from_patterns(normalized, patterns)
        if val is not None:
            return val, unit
    return None, None


def waiting_days_estimate(value: Optional[int], unit: Optional[str]) -> Optional[int]:
    if value is None or unit is None:
        return None
    unit_lower = unit.lower()
    if unit_lower in ("jour", "j+", "depuis"):
        return value
    if unit_lower == "semaine":
        return value * 7
    if unit_lower == "mois":
        return value * 30
    if unit_lower == "an":
        return value * 365
    if unit_lower == "heure":
        return max(1, value // 24)
    return value
