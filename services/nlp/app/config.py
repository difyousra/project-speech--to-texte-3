import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/app")).resolve()

XLM_ROBERTA_BASE_PATH = os.getenv(
    "XLM_ROBERTA_BASE_PATH",
    str(
        PROJECT_ROOT
        / "NLP/local_models/Roberta_model/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
    ),
)
MULTITASK_WEIGHTS_PATH = os.getenv(
    "MULTITASK_WEIGHTS_PATH",
    str(PROJECT_ROOT / "NLP/RESULT_multiTask/MultiModel_classifier.pt"),
)
MULTITASK_TOKENIZER_PATH = os.getenv(
    "MULTITASK_TOKENIZER_PATH",
    str(PROJECT_ROOT / "NLP/RESULT_multiTask/tokenizer2_0"),
)
DZIRIBERT_PATH = os.getenv(
    "DZIRIBERT_PATH",
    str(PROJECT_ROOT / "NLP/local_models/dziribert_local"),
)
LV1_MODEL_PATH = os.getenv(
    "LV1_MODEL_PATH",
    str(PROJECT_ROOT / "NLP/RESULT_monoTask/model_classifier_lv1"),
)
LV1_TOKENIZER_PATH = os.getenv(
    "LV1_TOKENIZER_PATH",
    str(PROJECT_ROOT / "NLP/RESULT_monoTask/tokenizer1"),
)

def _default_device() -> str:
    explicit = os.getenv("DEVICE", "").lower()
    if explicit in ("cpu", "cuda"):
        return explicit
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _default_device()
LOW_MEMORY = os.getenv("LOW_MEMORY", "false").lower() in ("1", "true", "yes")
API_KEY = os.getenv("API_KEY", "")
HOST = os.getenv("NLP_HOST", "0.0.0.0")
PORT = int(os.getenv("NLP_PORT", "8003"))
