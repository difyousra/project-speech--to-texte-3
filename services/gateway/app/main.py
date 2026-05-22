import asyncio
import io
import os
import socket
import sys
import uuid
from typing import Optional

import httpx
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

LIBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from reclamation_common.schemas import AnalyzeRequest  # noqa: E402

from app.config import (  # noqa: E402
    API_KEY,
    HOST,
    NLP_SERVICE_URL,
    PORT,
    STT_SERVICE_URL,
    VAD_SERVICE_URL,
)
from app.deepseek_advisor import generate_at_solution, is_advisor_enabled  # noqa: E402
from app.stats_store import get_dashboard_stats, record_reclamation  # noqa: E402
from app.ws_handler import handle_live_websocket  # noqa: E402

DEMO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../demo/static/index.html")
)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="Reclamation Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CERT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../certs/cert.pem")
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "microphone=(self)"
    return response


@app.get("/health")
async def health():
    status = {"gateway": "ok", "vad": "unknown", "stt": "unknown", "nlp": "unknown"}
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    async with httpx.AsyncClient() as client:
        for name, url in [
            ("vad", f"{VAD_SERVICE_URL}/health"),
            ("stt", f"{STT_SERVICE_URL}/health"),
            ("nlp", f"{NLP_SERVICE_URL}/health"),
        ]:
            try:
                if name == "stt":
                    r = await client.get(f"{STT_SERVICE_URL}/ready", headers=headers, timeout=5.0)
                    status[name] = "ok"
                else:
                    r = await client.get(url, headers=headers, timeout=5.0)
                    status[name] = "ok" if r.status_code == 200 else "error"
            except httpx.HTTPStatusError as e:
                status[name] = "loading" if name == "stt" and e.response.status_code == 503 else "error"
            except httpx.HTTPError:
                status[name] = "down"
    status["deepseek"] = "ok" if is_advisor_enabled() else "disabled"
    return status


@app.get("/api/v1/stats/dashboard")
async def stats_dashboard():
    return get_dashboard_stats()


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await handle_live_websocket(websocket)


@app.post("/api/v1/analyze")
async def analyze_proxy(
    body: AnalyzeRequest,
    _: None = Depends(verify_api_key),
):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{NLP_SERVICE_URL}/api/v1/analyze",
            json=body.model_dump(),
            headers=headers,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = body.session_id or str(uuid.uuid4())
        record_reclamation(
            session_id=session_id,
            transcript=body.text,
            classification=data.get("classification"),
            sentiment=data.get("sentiment"),
            source="analyze",
        )
        advisor = await generate_at_solution(
            body.text,
            classification=data.get("classification"),
            sentiment=data.get("sentiment"),
        )
        return {**data, "advisor": advisor}


class AdviseRequest(BaseModel):
    transcript: str
    language: str = "darija"
    classification: Optional[dict] = None
    sentiment: Optional[dict] = None


@app.post("/api/v1/advise")
async def advise(
    body: AdviseRequest,
    _: None = Depends(verify_api_key),
):
    return await generate_at_solution(
        body.transcript,
        classification=body.classification,
        sentiment=body.sentiment,
        language=body.language,
    )


@app.post("/api/v1/transcribe")
async def transcribe_proxy(
    file: UploadFile = File(...),
    language_hint: Optional[str] = None,
    _: None = Depends(verify_api_key),
):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    data = await file.read()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{STT_SERVICE_URL}/api/v1/transcribe",
            files={"file": (file.filename or "audio.wav", data, file.content_type or "audio/wav")},
            params={"language_hint": language_hint} if language_hint else {},
            headers=headers,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()


async def _post_stt(
    client: httpx.AsyncClient,
    seg_bytes: bytes,
    detected_lang: str,
    headers: dict,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            return await client.post(
                f"{STT_SERVICE_URL}/api/v1/transcribe",
                files={"file": ("seg.wav", seg_bytes, "audio/wav")},
                params={"language_hint": detected_lang, "partial": "false"},
                headers=headers,
                timeout=600.0,
            )
        except httpx.ConnectError as exc:
            last_exc = exc
            if attempt < 5:
                await asyncio.sleep(8)
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _service_error(service: str, exc: Exception) -> HTTPException:
    if isinstance(exc, httpx.ConnectError):
        if service.upper() == "STT":
            return HTTPException(
                status_code=503,
                detail=(
                    "Service STT indisponible ou en chargement (Whisper Darija, ~1 min). "
                    "Attendez puis réessayez, ou : docker compose up -d stt"
                ),
            )
        return HTTPException(
            status_code=503,
            detail=f"Service {service} indisponible. Lancez : docker compose up -d {service.lower()}",
        )
    if isinstance(exc, httpx.HTTPStatusError):
        body = exc.response.text[:500] if exc.response else str(exc)
        return HTTPException(
            status_code=502,
            detail=f"Erreur {service} ({exc.response.status_code}): {body}",
        )
    return HTTPException(status_code=500, detail=f"Erreur {service}: {exc}")


@app.post("/api/v1/pipeline")
async def pipeline(
    file: UploadFile = File(...),
    language_hint: Optional[str] = None,
    _: None = Depends(verify_api_key),
):
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Fichier audio vide")

    async with httpx.AsyncClient() as client:
        try:
            vad_resp = await client.post(
                f"{VAD_SERVICE_URL}/api/v1/segment",
                files={"file": (file.filename or "audio.wav", data, file.content_type or "audio/wav")},
                headers=headers,
                timeout=120.0,
            )
            vad_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _service_error("VAD", exc) from exc

        segments = vad_resp.json().get("segments", [])
        if not segments:
            return {
                "session_id": str(uuid.uuid4()),
                "transcript": "",
                "language": language_hint or "darija",
                "validated": False,
                "classification": None,
                "sentiment": None,
                "message": "Aucun segment vocal détecté dans l'audio",
            }

        transcripts = []
        detected_lang = language_hint or os.getenv("STT_DEFAULT_LANGUAGE", "darija")
        for seg in segments:
            seg_bytes = __import__("base64").b64decode(seg["audio_base64"])
            try:
                stt_resp = await _post_stt(client, seg_bytes, detected_lang, headers)
                stt_resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise _service_error("STT", exc) from exc

            stt_data = stt_resp.json()
            seg_text = (stt_data.get("text") or "").strip()
            if seg_text and (stt_data.get("validated") or len(seg_text) >= 8):
                transcripts.append(seg_text)
            detected_lang = stt_data.get("language", detected_lang)

        full_text = " ".join(transcripts).strip()
        if not full_text:
            return {
                "session_id": str(uuid.uuid4()),
                "transcript": "",
                "language": detected_lang,
                "validated": False,
                "classification": None,
                "sentiment": None,
                "message": "Transcription non validée ou vide — réessayez avec plus de parole",
            }

        try:
            nlp_resp = await client.post(
                f"{NLP_SERVICE_URL}/api/v1/analyze",
                json={"text": full_text},
                headers=headers,
                timeout=120.0,
            )
            nlp_resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise _service_error("NLP", exc) from exc

        nlp_data = nlp_resp.json()

    session_id = str(uuid.uuid4())
    record_reclamation(
        session_id=session_id,
        transcript=full_text,
        classification=nlp_data.get("classification"),
        sentiment=nlp_data.get("sentiment"),
        source="pipeline",
    )

    advisor = await generate_at_solution(
        full_text,
        classification=nlp_data.get("classification"),
        sentiment=nlp_data.get("sentiment"),
        language=detected_lang,
    )

    return {
        "session_id": session_id,
        "transcript": full_text,
        "language": detected_lang,
        "validated": True,
        "classification": nlp_data.get("classification"),
        "sentiment": nlp_data.get("sentiment"),
        "advisor": advisor,
    }


def _lan_ipv4_addresses() -> list[str]:
    """Adresses IPv4 privées utilisables depuis le LAN (hôte Docker)."""
    found: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                found.add(ip)
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            found.add(ip)
    except OSError:
        pass
    return sorted(found)


@app.get("/api/v1/network-info")
async def network_info():
    port = os.getenv("GATEWAY_PORT", "8000")
    https_port = os.getenv("GATEWAY_HTTPS_PORT", "8443")
    has_ssl = os.path.isfile(CERT_PATH)
    public = os.getenv("LAN_PUBLIC_URL", "").strip().rstrip("/")
    if public:
        urls = [public + "/"]
        if has_ssl and public.startswith("http://"):
            urls.insert(0, public.replace("http://", "https://").replace(f":{port}", f":{https_port}") + "/")
        return {
            "port": int(port),
            "https_port": int(https_port),
            "has_ssl": has_ssl,
            "secure_context_required": True,
            "public_url": urls[0],
            "suggested_urls": urls,
            "websocket_path": "/ws/live",
            "hint": "Sur téléphone : utilisez l’URL https:// (port 8443) pour autoriser le micro.",
        }
    ips = _lan_ipv4_addresses()
    suggested = [f"http://{ip}:{port}/" for ip in ips]
    return {
        "port": int(port),
        "lan_ips": ips,
        "suggested_urls": suggested,
        "websocket_path": "/ws/live",
        "hint": "Définissez LAN_PUBLIC_URL dans .env (IP Wi‑Fi du PC) si les liens sont incorrects.",
    }


@app.get("/")
@app.get("/demo")
async def demo_page():
    if os.path.isfile(DEMO_PATH):
        return FileResponse(DEMO_PATH)
    raise HTTPException(status_code=404, detail="Demo page not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
