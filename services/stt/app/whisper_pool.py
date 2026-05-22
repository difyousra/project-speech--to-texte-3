import logging
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from app.config import (
    DEVICE,
    LANG_MAP,
    MAX_LOADED_MODELS,
    SAMPLE_RATE,
    STT_MAX_NEW_TOKENS_FULL,
    STT_MAX_NEW_TOKENS_PARTIAL,
    STT_QUANTIZE_DYNAMIC,
    STT_USE_FASTER_WHISPER,
)

logger = logging.getLogger(__name__)

LanguageCode = Literal["fr", "ar", "darija"]
# ~30 s max pour Whisper small (1500 frames mel × stride)
MAX_SAMPLES = int(SAMPLE_RATE * 30)


class WhisperPool:
    def __init__(self):
        self._hf_models: OrderedDict[str, tuple[WhisperForConditionalGeneration, WhisperProcessor]] = (
            OrderedDict()
        )
        self._fw_models: OrderedDict[str, object] = OrderedDict()
        self._fw_failed: set[str] = set()
        self.device = "cuda" if torch.cuda.is_available() and DEVICE == "cuda" else "cpu"

    def _evict_if_needed(self, cache: OrderedDict) -> None:
        while len(cache) >= MAX_LOADED_MODELS:
            evicted_lang, _ = cache.popitem(last=False)
            logger.info("Evicted STT model for language %s", evicted_lang)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _load_hf(self, lang: LanguageCode) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
        if lang in self._hf_models:
            self._hf_models.move_to_end(lang)
            return self._hf_models[lang]

        path = LANG_MAP[lang]
        self._evict_if_needed(self._hf_models)
        processor = WhisperProcessor.from_pretrained(path)
        model = WhisperForConditionalGeneration.from_pretrained(path)
        model.to(self.device)
        model.eval()
        if STT_QUANTIZE_DYNAMIC and self.device == "cpu":
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Whisper %s: quantification dynamique CPU activée", lang)
            except Exception as exc:
                logger.warning("Quantification dynamique ignorée: %s", exc)
        self._hf_models[lang] = (model, processor)
        return model, processor

    def _load_faster(self, lang: LanguageCode):
        if lang in self._fw_failed:
            raise RuntimeError(f"faster-whisper unavailable for {lang}")
        if lang in self._fw_models:
            self._fw_models.move_to_end(lang)
            return self._fw_models[lang]

        from faster_whisper import WhisperModel

        path = LANG_MAP[lang]
        self._evict_if_needed(self._fw_models)
        compute_type = "float16" if self.device == "cuda" else "int8"
        try:
            model = WhisperModel(path, device=self.device, compute_type=compute_type)
        except Exception as exc:
            self._fw_failed.add(lang)
            logger.warning("faster-whisper load failed for %s: %s — fallback HF", lang, exc)
            raise
        self._fw_models[lang] = model
        return model

    @staticmethod
    def _trim_audio(audio: np.ndarray) -> np.ndarray:
        if len(audio) > MAX_SAMPLES:
            return audio[-MAX_SAMPLES:].copy()
        return audio

    def transcribe(
        self,
        audio: np.ndarray,
        lang: LanguageCode,
        partial: bool = False,
    ) -> tuple[str, float]:
        audio = self._trim_audio(audio)
        if STT_USE_FASTER_WHISPER and lang not in self._fw_failed:
            try:
                return self._transcribe_faster(audio, lang, partial)
            except Exception:
                logger.info("Fallback HF pour lang=%s", lang)
        return self._transcribe_hf(audio, lang, partial)

    def _transcribe_hf(
        self,
        audio: np.ndarray,
        lang: LanguageCode,
        partial: bool,
    ) -> tuple[str, float]:
        model, processor = self._load_hf(lang)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        forced_lang = "fr" if lang == "fr" else "ar"
        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=True,
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        max_new_tokens = STT_MAX_NEW_TOKENS_PARTIAL if partial else STT_MAX_NEW_TOKENS_FULL
        gen_kwargs: dict = {
            "language": forced_lang,
            "task": "transcribe",
            "max_new_tokens": max_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            predicted_ids = model.generate(input_features, **gen_kwargs)
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        conf = 0.9 if len(text) >= 8 else 0.75
        return text, conf

    def _transcribe_faster(
        self,
        audio: np.ndarray,
        lang: LanguageCode,
        partial: bool,
    ) -> tuple[str, float]:
        model = self._load_faster(lang)
        forced_lang = "fr" if lang == "fr" else "ar"
        max_len = STT_MAX_NEW_TOKENS_PARTIAL if partial else STT_MAX_NEW_TOKENS_FULL
        segments, info = model.transcribe(
            audio,
            language=forced_lang,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
            max_new_tokens=max_len,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        conf = float(getattr(info, "language_probability", 0.85) or 0.85)
        return text, conf
