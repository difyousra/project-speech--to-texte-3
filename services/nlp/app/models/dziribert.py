import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer


class DziriBertSentiment:
    LABELS = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax().item())
        return self.LABELS[idx], float(probs[idx].item())
