import torch
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer


class RobertaMultiTask(nn.Module):
    def __init__(self, model_path: str, num_labels_1: int, num_labels_2: int, init_weights: bool = True):
        super().__init__()
        config = XLMRobertaConfig.from_pretrained(model_path)
        self.roberta = XLMRobertaModel(config)
        if init_weights:
            # Chargement complet (entraînement / notebooks)
            pretrained = XLMRobertaModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
            )
            self.roberta.load_state_dict(pretrained.state_dict())
            del pretrained
        self.dropout = nn.Dropout(0.1)
        self.classifier1 = nn.Linear(768, num_labels_1)
        self.classifier2 = nn.Linear(768, num_labels_2)

    def forward(self, input_ids, attention_mask, label_1=None, label_2=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])
        logits1 = self.classifier1(pooled_output)
        logits2 = self.classifier2(pooled_output)

        loss = None
        if label_1 is not None and label_2 is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = 0.4 * loss_fct(logits1, label_1) + 0.6 * loss_fct(logits2, label_2)

        return {"loss": loss, "logits": (logits1, logits2)}


class MultiTaskClassifier:
    def __init__(
        self,
        base_path: str,
        weights_path: str,
        tokenizer_path: str,
        device: str,
        num_labels_1: int = 5,
        num_labels_2: int = 5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        # init_weights=False : une seule passe de chargement via le .pt (évite OOM)
        self.model = RobertaMultiTask(base_path, num_labels_1, num_labels_2, init_weights=False)
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, assign=True)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)

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
        outputs = self.model(**inputs)
        logits1, logits2 = outputs["logits"]

        probs1 = torch.softmax(logits1, dim=-1)[0]
        p1_idx = int(probs1.argmax().item())

        if p1_idx != 0:
            p2_idx = None
            conf = {"intent": float(probs1[p1_idx].item())}
        else:
            probs2 = torch.softmax(logits2, dim=-1)[0]
            p2_idx = int(probs2.argmax().item())
            conf = {
                "intent": float(probs1[p1_idx].item()),
                "category": float(probs2[p2_idx].item()),
            }

        return p1_idx, p2_idx, conf
