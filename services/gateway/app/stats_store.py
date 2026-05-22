"""Statistiques agrégées des réclamations (Redis)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
RECENT_MAX = 100
_KEY_TOTAL = "reclamation:stats:total"


def _client():
    import redis

    return redis.from_url(REDIS_URL, decode_responses=True)


def record_reclamation(
    *,
    session_id: str,
    transcript: str,
    classification: dict[str, Any] | None,
    sentiment: dict[str, Any] | None,
    source: str = "pipeline",
) -> None:
    if not classification:
        return
    try:
        r = _client()
        r.incr(_KEY_TOTAL)
        priority = (classification.get("priority") or "medium").lower()
        category = classification.get("category") or classification.get("level2_label") or "autre"
        intent = classification.get("intent") or "unknown"
        service = classification.get("service") or "unknown"
        sent = (sentiment or {}).get("sentiment") or "neutral"

        r.hincrby("reclamation:stats:by_priority", priority, 1)
        r.hincrby("reclamation:stats:by_category", category, 1)
        r.hincrby("reclamation:stats:by_intent", intent, 1)
        r.hincrby("reclamation:stats:by_service", service, 1)
        r.hincrby("reclamation:stats:by_sentiment", sent, 1)

        entry = {
            "session_id": session_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "transcript_preview": (transcript or "")[:120],
            "priority": priority,
            "category": category,
            "intent": intent,
            "service": service,
            "sentiment": sent,
            "level1_label": classification.get("level1_label"),
            "level2_label": classification.get("level2_label"),
        }
        r.lpush("reclamation:stats:recent", json.dumps(entry, ensure_ascii=False))
        r.ltrim("reclamation:stats:recent", 0, RECENT_MAX - 1)
    except Exception as exc:
        logger.warning("Stats Redis non enregistrées: %s", exc)


def get_dashboard_stats() -> dict[str, Any]:
    try:
        r = _client()
        total = int(r.get(_KEY_TOTAL) or 0)
        by_priority = r.hgetall("reclamation:stats:by_priority") or {}
        by_category = r.hgetall("reclamation:stats:by_category") or {}
        by_intent = r.hgetall("reclamation:stats:by_intent") or {}
        by_service = r.hgetall("reclamation:stats:by_service") or {}
        by_sentiment = r.hgetall("reclamation:stats:by_sentiment") or {}
        recent_raw = r.lrange("reclamation:stats:recent", 0, 19)
        recent = []
        for item in recent_raw:
            try:
                recent.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return {
            "total": total,
            "by_priority": {k: int(v) for k, v in by_priority.items()},
            "by_category": {k: int(v) for k, v in by_category.items()},
            "by_intent": {k: int(v) for k, v in by_intent.items()},
            "by_service": {k: int(v) for k, v in by_service.items()},
            "by_sentiment": {k: int(v) for k, v in by_sentiment.items()},
            "recent": recent,
        }
    except Exception as exc:
        logger.warning("Stats Redis lecture: %s", exc)
        return {
            "total": 0,
            "by_priority": {},
            "by_category": {},
            "by_intent": {},
            "by_service": {},
            "by_sentiment": {},
            "recent": [],
            "error": "Redis indisponible",
        }
