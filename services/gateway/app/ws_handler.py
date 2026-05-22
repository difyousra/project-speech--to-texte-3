import asyncio
import io
import json
import logging
import os
import uuid

import httpx
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.config import API_KEY, NLP_SERVICE_URL, STT_SERVICE_URL
from app.deepseek_advisor import generate_at_solution, is_advisor_enabled
from app.stats_store import record_reclamation

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BUFFER_SECONDS = float(os.getenv("WS_BUFFER_SECONDS", "2.0"))
DEFAULT_LANGUAGE = os.getenv("STT_DEFAULT_LANGUAGE", "darija")
MIN_AUDIO_SEC = float(os.getenv("WS_MIN_AUDIO_SEC", "1.0"))
MIN_RMS = float(os.getenv("WS_MIN_RMS", "0.005"))


class LiveSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.buffer = np.array([], dtype=np.float32)
        self.transcript_parts: list[str] = []
        self.language: str = "darija" if DEFAULT_LANGUAGE in ("darija", "dz") else DEFAULT_LANGUAGE
        self.stt_busy = False
        self._nlp_tasks: set[asyncio.Task] = set()
        self._stt_task: asyncio.Task | None = None

    def append_pcm(self, pcm_bytes: bytes) -> None:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, samples])

    def should_process(self) -> bool:
        return len(self.buffer) >= int(SAMPLE_RATE * max(BUFFER_SECONDS, MIN_AUDIO_SEC))

    def consume_buffer(self) -> np.ndarray:
        audio = self.buffer.copy()
        self.buffer = np.array([], dtype=np.float32)
        return audio


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _rejection_message(reason: str) -> str:
    messages = {
        "empty": "Aucun texte détecté",
        "too_short": "Texte trop court — continuez à parler",
        "too_few_words": "Pas assez de mots — reformulez",
        "hallucination": "Segment ignoré (bruit / non parole)",
        "repetition": "Segment ignoré (répétition)",
        "low_confidence": "Confiance trop faible — répétez",
        "no_speech_content": "Parole non reconnue — parlez plus fort",
    }
    return messages.get(reason, "Transcription non validée pour analyse")


async def _post_wav(client: httpx.AsyncClient, url: str, audio: np.ndarray, **params) -> dict:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    files = {"file": ("chunk.wav", buf.read(), "audio/wav")}
    resp = await client.post(url, files=files, params=params, headers=headers, timeout=600.0)
    resp.raise_for_status()
    return resp.json()


async def _send(websocket: WebSocket, session_id: str, msg_type: str, payload: dict) -> None:
    await websocket.send_json(
        {"type": msg_type, "session_id": session_id, "payload": payload}
    )


async def _run_nlp(
    websocket: WebSocket,
    client: httpx.AsyncClient,
    session: LiveSession,
    full_transcript: str,
) -> None:
    try:
        await _send(websocket, session.session_id, "status", {"phase": "processing_nlp"})
        nlp_resp = await client.post(
            f"{NLP_SERVICE_URL}/api/v1/analyze",
            json={"text": full_transcript, "session_id": session.session_id},
            headers={"X-API-Key": API_KEY} if API_KEY else {},
            timeout=120.0,
        )
        nlp_resp.raise_for_status()
        data = nlp_resp.json()
        record_reclamation(
            session_id=session.session_id,
            transcript=full_transcript,
            classification=data.get("classification"),
            sentiment=data.get("sentiment"),
            source="live",
        )
        await _send(
            websocket,
            session.session_id,
            "nlp_result",
            {
                "transcript": full_transcript,
                "classification": data.get("classification"),
                "sentiment": data.get("sentiment"),
            },
        )
        if is_advisor_enabled():
            await _send(
                websocket,
                session.session_id,
                "status",
                {"phase": "processing_advisor", "message": "Conseiller IA Algérie Télécom…"},
            )
            advisor = await generate_at_solution(
                full_transcript,
                classification=data.get("classification"),
                sentiment=data.get("sentiment"),
                language=session.language,
            )
            await _send(
                websocket,
                session.session_id,
                "advisor_result",
                advisor,
            )
    except Exception as exc:
        logger.exception("NLP failed")
        await _send(
            websocket,
            session.session_id,
            "error",
            {"message": f"Analyse NLP: {exc}"},
        )


async def _process_audio_chunk(
    websocket: WebSocket,
    client: httpx.AsyncClient,
    session: LiveSession,
    audio: np.ndarray,
) -> None:
    duration_sec = len(audio) / SAMPLE_RATE
    energy = _rms(audio)

    if duration_sec < MIN_AUDIO_SEC:
        return

    if energy < MIN_RMS:
        await _send(
            websocket,
            session.session_id,
            "stt_skip",
            {"reason": "silence", "message": "Segment trop silencieux — parlez plus fort"},
        )
        return

    await _send(
        websocket,
        session.session_id,
        "status",
        {
            "phase": "processing_stt",
            "message": "Transcription Darija en cours (≈20–40 s sur CPU)…",
        },
    )

    stt_result = await _post_wav(
        client,
        f"{STT_SERVICE_URL}/api/v1/transcribe",
        audio,
        language_hint=session.language,
        partial="true",
        session_id=session.session_id,
    )

    text = (stt_result.get("text") or "").strip()
    session.language = stt_result.get("language", session.language)
    validated = bool(stt_result.get("validated"))

    if not text:
        await _send(
            websocket,
            session.session_id,
            "stt_empty",
            {
                "message": "Aucun texte détecté — réessayez en parlant plus clairement",
                "language": session.language,
            },
        )
        return

    if not validated:
        reason = stt_result.get("validation_reason") or "invalid"
        await _send(
            websocket,
            session.session_id,
            "transcript_rejected",
            {
                "text": text,
                "reason": reason,
                "message": _rejection_message(reason),
                "language": session.language,
            },
        )
        return

    session.transcript_parts.append(text)
    full_transcript = " ".join(session.transcript_parts)

    await _send(
        websocket,
        session.session_id,
        "partial_transcript",
        {
            **stt_result,
            "text": text,
            "full_transcript": full_transcript,
            "validated": True,
        },
    )

    task = asyncio.create_task(_run_nlp(websocket, client, session, full_transcript))
    session._nlp_tasks.add(task)
    task.add_done_callback(session._nlp_tasks.discard)


async def _run_stt_job(
    websocket: WebSocket,
    client: httpx.AsyncClient,
    session: LiveSession,
    audio: np.ndarray,
) -> None:
    try:
        await _process_audio_chunk(websocket, client, session, audio)
    except httpx.HTTPError as exc:
        logger.exception("STT failed")
        await _send(
            websocket,
            session.session_id,
            "error",
            {"message": f"Transcription: {exc}"},
        )
    finally:
        session.stt_busy = False
        session._stt_task = None
        if session.should_process() and not session.stt_busy:
            await schedule_stt_if_ready(websocket, client, session)


async def schedule_stt_if_ready(
    websocket: WebSocket,
    client: httpx.AsyncClient,
    session: LiveSession,
) -> None:
    if session.stt_busy or not session.should_process():
        return
    session.stt_busy = True
    audio = session.consume_buffer()
    session._stt_task = asyncio.create_task(_run_stt_job(websocket, client, session, audio))


async def handle_live_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    session = LiveSession(session_id)

    await _send(
        websocket,
        session_id,
        "session_started",
        {
            "sample_rate": SAMPLE_RATE,
            "buffer_seconds": BUFFER_SECONDS,
            "hint": "Parlez 3–5 s puis attendez ~1 min : le texte s’affiche segment par segment",
        },
    )

    async with httpx.AsyncClient() as client:
        try:
            while True:
                message = await websocket.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"]:
                    session.append_pcm(message["bytes"])
                    await schedule_stt_if_ready(websocket, client, session)

                elif "text" in message and message["text"]:
                    data = json.loads(message["text"])
                    if data.get("type") == "end_call":
                        if session.buffer.size > 0 and not session.stt_busy:
                            await schedule_stt_if_ready(websocket, client, session)
                        if session._stt_task:
                            await session._stt_task
                        if session._nlp_tasks:
                            await asyncio.gather(*session._nlp_tasks, return_exceptions=True)
                        full = " ".join(session.transcript_parts).strip()
                        if full:
                            await _run_nlp(websocket, client, session, full)
                        await _send(
                            websocket,
                            session_id,
                            "final_transcript",
                            {"transcript": full},
                        )
                        break

        except WebSocketDisconnect:
            pass
