"""Conseiller réclamation Algérie Télécom via API DeepSeek (compatible OpenAI)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"
).strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
DEEPSEEK_ENABLED = os.getenv("DEEPSEEK_ENABLED", "true").lower() in ("1", "true", "yes")
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "90"))


def is_advisor_enabled() -> bool:
    return DEEPSEEK_ENABLED and bool(DEEPSEEK_API_KEY)


def _format_classification(c: dict[str, Any] | None) -> str:
    if not c:
        return "Non disponible"
    lines = [
        f"Intention : {c.get('intent')}",
        f"Catégorie : {c.get('level2_label') or c.get('category')}",
        f"Priorité : {c.get('priority')}",
        f"Urgence : {'oui' if c.get('urgency') else 'non'}",
        f"Service chargé : {c.get('service_label') or c.get('service')}",
        f"SLA prise en charge : {c.get('handling_sla_days')} jour(s) ouvré(s)"
        if c.get("handling_sla_days")
        else "",
        f"Délai signalé client : {c.get('waiting_days')} j ({c.get('waiting_unit')})"
        if c.get("waiting_days")
        else "",
    ]
    return "\n".join(l for l in lines if l)


def _format_sentiment(s: dict[str, Any] | None) -> str:
    if not s:
        return "Non disponible"
    score = s.get("score")
    pct = f"{float(score) * 100:.1f} %" if score is not None else "—"
    return f"{s.get('sentiment')} — émotion {s.get('emotion')} — confiance {pct}"


def _build_messages(
    transcript: str,
    classification: dict[str, Any] | None,
    sentiment: dict[str, Any] | None,
    language: str = "darija",
) -> list[dict[str, str]]:
    system = """Tu es un conseiller senior du service réclamations d'Algérie Télécom.
Tu réponds en français clair (tu peux citer des mots en darija de la transcription si utile).
À partir de la transcription client et de l'analyse automatique (classification, sentiment, priorité),
tu dois :
1. Reformuler brièvement le problème du client.
2. Proposer une solution concrète et réaliste du point de vue Algérie Télécom (étapes, délais, contacts 12/100/1500, espace client, technicien, facturation, activation ligne, etc.).
3. Indiquer le service interne chargé du dossier et le délai de prise en charge attendu.
4. Rester professionnel, empathique si le client est frustré, sans promesses impossibles.
Structure ta réponse avec des titres courts : Problème | Solution proposée | Prochaines étapes | Service concerné."""

    user = f"""Langue détectée : {language}

--- Transcription (appel client) ---
{transcript.strip()}

--- Classification automatique ---
{_format_classification(classification)}

--- Sentiment ---
{_format_sentiment(sentiment)}

Rédige la réponse conseiller pour traiter cette réclamation."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def generate_at_solution(
    transcript: str,
    classification: dict[str, Any] | None = None,
    sentiment: dict[str, Any] | None = None,
    language: str = "darija",
) -> dict[str, Any]:
    if not is_advisor_enabled():
        return {
            "enabled": False,
            "solution": None,
            "message": "Conseiller IA désactivé (DEEPSEEK_API_KEY manquante)",
        }

    if not (transcript or "").strip():
        return {"enabled": True, "solution": None, "message": "Transcription vide"}

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": _build_messages(transcript, classification, sentiment, language),
        "temperature": 0.4,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                DEEPSEEK_API_URL,
                json=payload,
                headers=headers,
                timeout=DEEPSEEK_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        solution = (message.get("content") or "").strip()
        usage = data.get("usage") or {}
        return {
            "enabled": True,
            "solution": solution,
            "model": DEEPSEEK_MODEL,
            "tokens": usage,
        }
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:400] if exc.response else str(exc)
        logger.exception("DeepSeek HTTP error")
        return {
            "enabled": True,
            "solution": None,
            "error": f"DeepSeek ({exc.response.status_code}): {body}",
        }
    except Exception as exc:
        logger.exception("DeepSeek call failed")
        return {"enabled": True, "solution": None, "error": str(exc)}
