"""Classifieur léger : un seul chargement HF (évite OOM du multi-tâche custom)."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LightweightClassifier:
    """Niveau 1 uniquement via model_classifier_lv1 (safetensors)."""

    ID2LABEL = {
        0: "Problème / Réclamation",
        1: "Inapproprié",
        2: "Question/Assistance",
        3: "Positif",
        4: "Suggestion",
    }

    KEYWORD_LEVEL2 = {
        "fibre": (1, "Déploiement de la fibre"),
        "فيبر": (1, "Déploiement de la fibre"),
        "panne": (0, "Pannes et coupures générales"),
        "coupure": (0, "Pannes et coupures générales"),
        "انقطاع": (0, "Pannes et coupures générales"),
        "installation": (2, "Retards d\u2019installation ou de mise en service"),
        "مودام": (2, "Retards d\u2019installation ou de mise en service"),
        "modem": (2, "Retards d\u2019installation ou de mise en service"),
        "idoom": (4, "Espace Client + My Idoom"),
        "my idoom": (4, "Espace Client + My Idoom"),
    }

    def __init__(self, model_path: str, tokenizer_path: str, device: str):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def _guess_level2(self, text: str) -> int | None:
        lowered = text.lower()
        for keyword, (idx, _) in self.KEYWORD_LEVEL2.items():
            if keyword in lowered:
                return idx
        return 0

    @torch.inference_mode()
    def predict(self, text: str) -> tuple[int, int | None, dict[str, float]]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        p1_idx = int(probs.argmax().item())
        conf = {"intent": float(probs[p1_idx].item())}

        if p1_idx != 0:
            return p1_idx, None, conf

        p2_idx = self._guess_level2(text)
        conf["category"] = 0.6
        return p1_idx, p2_idx, conf
