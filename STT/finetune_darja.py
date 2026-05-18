import os
import re
import csv
import io
import sys
import argparse
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import evaluate
import librosa
import soundfile as sf

from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def normalize_arabic_light(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("ـ", "")
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    return text


def normalize_metric_text(text: str, strip_accents: bool = False) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = normalize_arabic_light(text)

    if strip_accents:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = re.sub(r"[،,.!?؟؛:;\"“”()\[\]{}<>|/\\_=+*~^@#$%&]+", " ", text)
    text = re.sub(r"[-—–]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def resolve_audio_path(path: str, dataset_root: str = "dataset") -> str:
    path = str(path).strip().strip('"')
    if os.path.exists(path):
        return path
    candidate = os.path.join(dataset_root, path)
    if os.path.exists(candidate):
        return candidate
    return path


def read_csv_header(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def validate_csv(csv_path: str, audio_col: str, text_col: str, dataset_root: str, max_audio_seconds: float):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    header = read_csv_header(csv_path)
    missing = [c for c in [audio_col, text_col] if c not in header]
    if missing:
        raise KeyError(f"Colonnes manquantes dans {csv_path}: {missing}. Colonnes trouvées: {header}")

    missing_audio, empty_text, too_long = [], [], []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            audio_path = resolve_audio_path(row[audio_col], dataset_root)
            text = (row[text_col] or "").strip()

            if not text:
                empty_text.append(idx)

            if not os.path.exists(audio_path):
                missing_audio.append((idx, audio_path))
                continue

            try:
                info = sf.info(audio_path)
                dur = info.frames / float(info.samplerate)
                if dur > max_audio_seconds + 0.5:
                    too_long.append((idx, audio_path, round(dur, 2)))
            except Exception as e:
                missing_audio.append((idx, f"{audio_path} | unreadable: {e}"))

    errors = []
    if missing_audio:
        preview = "\n".join([f"ligne {i}: {p}" for i, p in missing_audio[:10]])
        errors.append(f"Audios introuvables/illisibles ({len(missing_audio)}):\n{preview}")
    if empty_text:
        errors.append(f"Textes vides ({len(empty_text)}), exemples lignes: {empty_text[:20]}")
    if too_long:
        preview = "\n".join([f"ligne {i}: {p} ({d}s)" for i, p, d in too_long[:10]])
        errors.append(f"Audios trop longs ({len(too_long)}):\n{preview}")

    if errors:
        raise ValueError("\n\n".join(errors))


def load_audio(path: str, target_sr: int, normalize_audio: bool, trim_silence: bool, silence_top_db: float):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if trim_silence and len(audio) > 0:
        audio, _ = librosa.effects.trim(audio, top_db=max(1.0, float(silence_top_db)))

    if normalize_audio and len(audio) > 0:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak

    if len(audio) == 0:
        raise ValueError(f"Audio vide après preprocessing: {path}")

    return audio, sr


def parse_args():
    p = argparse.ArgumentParser("Fine-tuning Whisper Darija code-switching AR/FR")

    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--train_csv", default=r"laststation/dataset_augmented.csv")
    p.add_argument("--val_csv", default=r"laststation/validation.csv")
    p.add_argument("--output_dir", default=r"output_darja")
    p.add_argument("--dataset_root", default="laststation")

    p.add_argument("--audio_column", default="audio_path")
    p.add_argument("--text_column", default="transcription")

    p.add_argument("--language", default="ar")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    p.add_argument("--target_sampling_rate", type=int, default=16000)
    p.add_argument("--max_audio_seconds", type=float, default=30.0)
    p.add_argument("--max_label_length", type=int, default=256)
    p.add_argument("--normalize_audio", action="store_true")
    p.add_argument("--trim_silence", action="store_true")
    p.add_argument("--silence_top_db", type=float, default=35.0)

    p.add_argument("--num_train_epochs", type=float, default=7.0)
    p.add_argument("--learning_rate", type=float, default=8e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_type", default="linear")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # IMPORTANT: 0.0 évite le bug decoder_input_ids + decoder_inputs_embeds avec Whisper
    p.add_argument("--label_smoothing_factor", type=float, default=0.0)

    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--eval_strategy", default="steps", choices=["steps", "epoch", "no"])
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_strategy", default="steps", choices=["steps", "epoch"])
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--early_stopping_patience", type=int, default=5)

    p.add_argument("--generation_max_length", type=int, default=225)
    p.add_argument("--generation_num_beams", type=int, default=1)
    p.add_argument("--strip_accents_for_metrics", action="store_true")

    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--report_to", default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", default="")
    p.add_argument("--debug_samples", type=int, default=5)

    return p.parse_args()


def main():
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    print(f"[INFO] CUDA: {use_cuda}")
    if use_cuda:
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    if args.tf32 and use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    has_val = bool(args.val_csv) and os.path.exists(args.val_csv)

    validate_csv(args.train_csv, args.audio_column, args.text_column, args.dataset_root, args.max_audio_seconds)
    if has_val:
        validate_csv(args.val_csv, args.audio_column, args.text_column, args.dataset_root, args.max_audio_seconds)
    else:
        print("[WARN] Pas de validation CSV valide. Training sans WER/CER pendant l'entraînement.")

    data_files = {"train": args.train_csv}
    if has_val:
        data_files["validation"] = args.val_csv

    raw = load_dataset("csv", data_files=data_files)

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.use_cache = False

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.num_beams = args.generation_num_beams
    model.generation_config.max_length = args.generation_max_length

    if args.freeze_encoder:
        print("[INFO] Encoder frozen.")
        model.freeze_encoder()

    if args.gradient_checkpointing:
        print("[INFO] Gradient checkpointing enabled.")
        model.gradient_checkpointing_enable()

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = resolve_audio_path(batch[args.audio_column], args.dataset_root)
        text = str(batch[args.text_column]).strip()
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"\s+", " ", text).strip()

        audio, _ = load_audio(
            audio_path,
            target_sr=args.target_sampling_rate,
            normalize_audio=args.normalize_audio,
            trim_silence=args.trim_silence,
            silence_top_db=args.silence_top_db,
        )

        max_samples = int(args.max_audio_seconds * args.target_sampling_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        input_features = processor.feature_extractor(
            audio,
            sampling_rate=args.target_sampling_rate,
            return_attention_mask=False,
        ).input_features[0]

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

    train_dataset = raw["train"].filter(
        lambda x: x[args.audio_column] is not None and str(x[args.audio_column]).strip() != ""
    )
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
        num_proc=1,
        desc="Preprocessing train",
        load_from_cache_file=False,
    )

    eval_dataset = None
    if has_val:
        eval_dataset = raw["validation"].filter(
            lambda x: x[args.audio_column] is not None and str(x[args.audio_column]).strip() != ""
        )
        eval_dataset = eval_dataset.map(
            preprocess,
            remove_columns=eval_dataset.column_names,
            num_proc=1,
            desc="Preprocessing validation",
            load_from_cache_file=False,
        )

    print(f"[INFO] Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"[INFO] Validation samples: {len(eval_dataset)}")

    for i in range(min(args.debug_samples, len(train_dataset))):
        s = train_dataset[i]
        decoded = processor.tokenizer.decode(s["labels"], skip_special_tokens=True)
        print("\n[DEBUG SAMPLE]", i)
        print("decode:", decoded)

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

        pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_texts = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_norm = [normalize_metric_text(x, args.strip_accents_for_metrics) for x in pred_texts]
        label_norm = [normalize_metric_text(x, args.strip_accents_for_metrics) for x in label_texts]
        label_norm = [x if x else " " for x in label_norm]

        wer = wer_metric.compute(predictions=pred_norm, references=label_norm)
        cer = cer_metric.compute(predictions=pred_norm, references=label_norm)

        print("\n[METRIC DEBUG]")
        for i in range(min(5, len(pred_texts))):
            print(f"--- {i} ---")
            print("PRED RAW :", pred_texts[i])
            print("REAL RAW :", label_texts[i])
            print("PRED NORM:", pred_norm[i])
            print("REAL NORM:", label_norm[i])

        return {"wer": wer, "cer": cer}

    report_to = [x.strip() for x in args.report_to.split(",") if x.strip()] or ["none"]
    eval_strategy = args.eval_strategy if eval_dataset is not None else "no"
    save_strategy = args.save_strategy if eval_dataset is not None else "epoch"

    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else 0
    warmup_ratio = 0.0 if warmup_steps > 0 else args.warmup_ratio

    training_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        report_to=report_to,
        save_strategy=save_strategy,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="wer" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        predict_with_generate=eval_dataset is not None,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        remove_unused_columns=False,
        label_smoothing_factor=0.0,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=args.max_grad_norm,
    )

    try:
        training_args = Seq2SeqTrainingArguments(
            **training_kwargs,
            eval_strategy=eval_strategy,
            eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        )
    except TypeError:
        training_args = Seq2SeqTrainingArguments(
            **training_kwargs,
            evaluation_strategy=eval_strategy,
            eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        )

    callbacks = []
    if eval_dataset is not None and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=callbacks,
        processing_class=processor,
    )

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            resume_checkpoint = get_last_checkpoint(args.output_dir)
        else:
            resume_checkpoint = args.resume_from_checkpoint
        print("[INFO] Resume checkpoint:", resume_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print("\n[INFO] Training terminé.")
    print("[INFO] Modèle sauvegardé dans:", args.output_dir)
    print("[INFO] Train metrics:", train_result.metrics)

    if eval_dataset is not None:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        print("[INFO] Final eval metrics:", metrics)


if __name__ == "__main__":
    main()