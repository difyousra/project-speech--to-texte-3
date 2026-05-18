import os
import re
import csv
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import io
import sys

import numpy as np
import torch
import evaluate
import librosa
import soundfile as sf

from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


# ============================================================
# Defaults
# ============================================================
DEFAULT_MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-small")
DEFAULT_DATASET_CSV = os.environ.get("WHISPER_DATASET_CSV", "dataset_darja\FINAL\dataset_augmented.csv")
DEFAULT_VALIDATION_CSV = os.environ.get("WHISPER_VALIDATION_CSV", "dataset/processed/validation.csv")
DEFAULT_OUTPUT_DIR = os.environ.get("WHISPER_OUTPUT_DIR", "whisper-finetuned-clean")

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


# ============================================================
# Text normalization for metrics
# ============================================================
PUNCT_TO_SPACE = r"[\.,;:!\?\-\—\_\(\)\[\]\{\}\"'`“”’…/\\|]+"


def normalize_metric_text(text: str) -> str:
    """Normalize text consistently for WER/CER evaluation.

    This is intentionally conservative for French ASR:
    - lowercase
    - unify apostrophes/quotes
    - remove punctuation noise
    - collapse whitespace
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(PUNCT_TO_SPACE, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Robust Whisper fine-tuning script")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_csv", type=str, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--validation_csv", type=str, default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")

    parser.add_argument("--language", type=str, default="ar")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])

    parser.add_argument("--target_sampling_rate", type=int, default=16000)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--max_label_length", type=int, default=256)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--silence_top_db", type=float, default=35.0)
    parser.add_argument("--normalize_audio", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["steps", "epoch", "no"])
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_strategy", type=str, default="steps", choices=["steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--logging_first_step", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    parser.add_argument("--generation_num_beams", type=int, default=1)
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--predict_with_generate", action="store_true")

    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--debug_samples", type=int, default=3)
    parser.add_argument("--do_lower_case_labels", action="store_true")

    return parser.parse_args()


# ============================================================
# CSV / audio validation
# ============================================================

def _read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV vide: {csv_path}")
    return rows
def _resolve_audio_path(audio_value: str) -> str:
    audio_path = audio_value.strip()
    if not audio_path:
        return ""
    if os.path.exists(audio_path):
        return audio_path
    if not audio_path.startswith("dataset/"):
        prefixed = os.path.join("dataset", audio_path)
        if os.path.exists(prefixed):
            return prefixed
        return prefixed
    return audio_path
def validate_dataset_audio(csv_path: str) -> None:
    rows = _read_csv_rows(csv_path)

    required_columns = {"audio", "text"}
    got = set(rows[0].keys())
    missing = required_columns - got
    if missing:
        raise KeyError(f"Colonnes manquantes dans {csv_path}: {sorted(missing)}")

    missing_audio = []
    empty_text = []
    too_long = []

    for idx, row in enumerate(rows, start=2):
        audio_path = _resolve_audio_path(row["audio"])
        text = (row.get("text") or "").strip()

        if not audio_path or not os.path.exists(audio_path):
            missing_audio.append((idx, audio_path))
        if not text:
            empty_text.append(idx)

        if os.path.exists(audio_path):
            try:
                info = sf.info(audio_path)
                dur = info.frames / float(info.samplerate)
                if dur > 30.5:
                    too_long.append((idx, audio_path, round(dur, 2)))
            except Exception:
                pass

    errors = []
    if missing_audio:
        preview = "\n".join([f"ligne {i}: {p}" for i, p in missing_audio[:10]])
        errors.append(f"Fichiers audio introuvables ({len(missing_audio)}):\n{preview}")
    if empty_text:
        preview = ", ".join(map(str, empty_text[:20]))
        errors.append(f"Textes vides ({len(empty_text)}), lignes: {preview}")
    if too_long:
        preview = "\n".join([f"ligne {i}: {p} ({d}s)" for i, p, d in too_long[:10]])
        errors.append(f"Audios > 30s détectés ({len(too_long)}):\n{preview}")

    if errors:
        raise ValueError("\n\n".join(errors))


# ============================================================
# Audio loading / preprocessing
# ============================================================

def load_audio(audio_path: str, target_sr: int, trim_silence: bool, silence_top_db: float, normalize_audio: bool):
    audio_array, sr = sf.read(audio_path)

    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)

    audio_array = audio_array.astype(np.float32)

    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if trim_silence and len(audio_array) > 0:
        audio_array, _ = librosa.effects.trim(audio_array, top_db=max(1.0, float(silence_top_db)))

    if normalize_audio and len(audio_array) > 0:
        peak = float(np.max(np.abs(audio_array)))
        if peak > 0:
            audio_array = audio_array / peak

    if len(audio_array) == 0:
        raise ValueError(f"Audio vide après preprocessing: {audio_path}")

    return audio_array, sr


# ============================================================
# Data collator
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS if it was appended everywhere by the tokenizer
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ============================================================
# Main
# ============================================================

def main():
    # Force la sortie standard à utiliser l'UTF-8 pour éviter les erreurs d'affichage
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    args = parse_args()
    set_seed(args.seed)

    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.output_dir, "runs")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] Device: {device}")

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] GPU: {gpu_name}")

    if not os.path.exists(args.dataset_csv):
        raise FileNotFoundError(f"Dataset train introuvable: {args.dataset_csv}")

    has_eval_file = bool(args.validation_csv) and os.path.exists(args.validation_csv)
    if args.validation_csv and not has_eval_file:
        print(f"[WARN] Validation CSV introuvable: {args.validation_csv}. L'évaluation sera désactivée.")

    validate_dataset_audio(args.dataset_csv)
    if has_eval_file:
        validate_dataset_audio(args.validation_csv)

    data_files = {"train": args.dataset_csv}
    if has_eval_file:
        data_files["validation"] = args.validation_csv

    raw_dataset = load_dataset("csv", data_files=data_files)

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False  # Ajoute cette ligne impérativement

    # Force language/task prompts for generation consistency
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.num_beams = args.generation_num_beams
    model.generation_config.max_length = args.generation_max_length

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)
    print(f"[INFO] Model loaded: {args.model}")

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = _resolve_audio_path(batch["audio"])
        text = (batch["text"] or "").strip()

        if args.do_lower_case_labels:
            text = text.lower()

        audio_array, sr = load_audio(
            audio_path=audio_path,
            target_sr=args.target_sampling_rate,
            trim_silence=args.trim_silence,
            silence_top_db=args.silence_top_db,
            normalize_audio=args.normalize_audio,
        )

        max_len_samples = int(args.max_audio_seconds * args.target_sampling_rate)
        if len(audio_array) > max_len_samples:
            audio_array = audio_array[:max_len_samples]

        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_attention_mask=False,
        ).input_features[0]

        # IMPORTANT: DO NOT disable special tokens here.
        labels = processor.tokenizer(
            text,
            max_length=args.max_label_length,
            truncation=True,
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "text": text,
            "audio_path": audio_path,
        }

    train_dataset = raw_dataset["train"]
    train_dataset = train_dataset.filter(lambda x: (x["audio"] or "").strip() != "")
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
        num_proc=4,
        desc="Preprocessing train dataset",
    )

    eval_dataset = None
    has_eval = False
    if "validation" in raw_dataset:
        eval_dataset = raw_dataset["validation"]
        eval_dataset = eval_dataset.filter(lambda x: (x["audio"] or "").strip() != "")
        eval_dataset = eval_dataset.map(
            preprocess,
            remove_columns=eval_dataset.column_names,
            num_proc=1,
            desc="Preprocessing validation dataset",
        )
        has_eval = len(eval_dataset) > 0

    print(f"[INFO] Train samples: {len(train_dataset)}")
    if has_eval:
        print(f"[INFO] Eval samples: {len(eval_dataset)}")

    # Sanity check on a few samples
    debug_n = min(args.debug_samples, len(train_dataset))
    for i in range(debug_n):
        sample = train_dataset[i]
        decoded_labels = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True)
        print("\n[DEBUG SAMPLE]", i)
        print("audio_path:", sample["audio_path"])
        print("label_text:", sample["text"])
        print("decoded_labels:", decoded_labels)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str_norm = [normalize_metric_text(s) for s in pred_str]
        label_str_norm = [normalize_metric_text(s) for s in label_str]
        label_str_norm = [s if s else " " for s in label_str_norm]

        wer = wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)
        cer = cer_metric.compute(predictions=pred_str_norm, references=label_str_norm)

        if len(pred_str) > 0:
            print("\n[METRIC DEBUG]")
            print("raw_pred   :", pred_str[0])
            print("raw_label  :", label_str[0])
            print("norm_pred  :", pred_str_norm[0])
            print("norm_label :", label_str_norm[0])

        return {"wer": wer, "cer": cer}

    report_to_value = [x.strip() for x in args.report_to.split(",") if x.strip()] or ["none"]
    eval_strategy = args.evaluation_strategy if has_eval else "no"
    save_strategy = args.save_strategy if has_eval else "epoch"

    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else 0
    warmup_ratio = 0.0 if warmup_steps > 0 else args.warmup_ratio

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=5,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        report_to=report_to_value,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps if (has_eval and eval_strategy == "steps") else None,
        save_strategy=save_strategy,
        save_steps=args.save_steps if (has_eval and save_strategy == "steps") else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model="wer" if has_eval else None,
        greater_is_better=False if has_eval else None,
        predict_with_generate=args.predict_with_generate or has_eval,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        remove_unused_columns=False,
        label_smoothing_factor=args.label_smoothing_factor,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        seed=args.seed,
        data_seed=args.seed,
        disable_tqdm=args.disable_tqdm,
        max_grad_norm=args.max_grad_norm,
    )

    callbacks = []
    if has_eval and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
        compute_metrics=compute_metrics if has_eval else None,
        callbacks=callbacks,
    )

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            resume_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_checkpoint:
                print(f"[INFO] Resume from checkpoint: {resume_checkpoint}")
            else:
                print("[INFO] No checkpoint found, starting from base model.")
        else:
            if not os.path.isdir(args.resume_from_checkpoint):
                raise FileNotFoundError(f"Checkpoint introuvable: {args.resume_from_checkpoint}")
            resume_checkpoint = args.resume_from_checkpoint
            print(f"[INFO] Resume from checkpoint: {resume_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\n[INFO] Training completed.")
    print("[INFO] Model saved to:", args.output_dir)
    print("[INFO] Train metrics:", train_result.metrics)

    if has_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        print("[INFO] Final eval metrics:", metrics)


if __name__ == "__main__":
    main()