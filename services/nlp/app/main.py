import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException

LIBS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
if LIBS_PATH not in sys.path:
    sys.path.insert(0, LIBS_PATH)

from reclamation_common.schemas import AnalyzeRequest, AnalyzeResponse  # noqa: E402

from app.config import API_KEY, HOST, PORT  # noqa: E402
from app.pipeline import analyze_text  # noqa: E402


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import torch

    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "4"))))
    logging.getLogger("nlp").info(
        "NLP prêt (DEVICE=%s, LOW_MEMORY=%s)", os.getenv("DEVICE"), os.getenv("LOW_MEMORY")
    )
    yield


app = FastAPI(title="NLP Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "nlp"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(
    body: AnalyzeRequest,
    _: None = Depends(verify_api_key),
):
    return analyze_text(body.text, body.session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
