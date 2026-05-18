import argparse
import csv
import logging
import os
import re
from typing import List, Tuple

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DEFAULT_MODEL_DIR = "whisper-finetuned"
DEFAULT_DATASET_CSV = os.environ.get("WHISPER_EVAL_DATASET_CSV", "dataset/processed/test.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("evaluate_whisper")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate finetuned Whisper with WER/CER")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to finetuned model directory")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_CSV, help="CSV file with audio,text columns")
    parser.add_argument("--max-samples", type=int, default=0, help="Evaluate only first N samples (0 = all)")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search width for generation")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein_distance(seq_a: List[str], seq_b: List[str]) -> int:
    len_a = len(seq_a)
    len_b = len(seq_b)

    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev_row = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr_row = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,
                curr_row[j - 1] + 1,
                prev_row[j - 1] + cost,
            )
        prev_row = curr_row
    return prev_row[len_b]


def word_error_rate(reference: str, hypothesis: str) -> Tuple[int, int, float]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    edits = levenshtein_distance(ref_words, hyp_words)
    total = max(1, len(ref_words))
    return edits, total, edits / total


def char_error_rate(reference: str, hypothesis: str) -> Tuple[int, int, float]:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    edits = levenshtein_distance(ref_chars, hyp_chars)
    total = max(1, len(ref_chars))
    return edits, total, edits / total


def load_rows(csv_path: str):
    rows = []
    skipped_missing_audio = 0
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "audio" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise KeyError("Le CSV doit contenir les colonnes 'audio' et 'text'.")
        for row in reader:
            audio_path = row["audio"].strip()
            text = row["text"].strip()
            if not audio_path.startswith("dataset/"):
                audio_path = os.path.join("dataset", audio_path)
            if not os.path.exists(audio_path):
                skipped_missing_audio += 1
                continue
            rows.append((audio_path, text))
    if skipped_missing_audio > 0:
        LOGGER.warning("%s fichiers audio absents ignores durant l'evaluation", skipped_missing_audio)
    return rows


def transcribe_file(processor, model, audio_path: str, device: str, num_beams: int) -> str:
    speech_array, sampling_rate = sf.read(audio_path)
    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    features = processor.feature_extractor(
        speech_array,
        sampling_rate=sampling_rate,
        return_attention_mask=True,
        return_tensors="pt",
    )

    model_dtype = next(model.parameters()).dtype
    input_features = features.input_features.to(device=device, dtype=model_dtype)
    attention_mask = None
    if hasattr(features, "attention_mask") and features.attention_mask is not None:
        attention_mask = features.attention_mask.to(device=device)

    with torch.no_grad():
        generate_kwargs = {
            "language": "fr",
            "task": "transcribe",
            "max_length": 128,
            "num_beams": max(1, num_beams),
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        predicted_ids = model.generate(input_features, **generate_kwargs)

    return processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    dataset_csv = os.path.abspath(args.dataset)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    rows = load_rows(dataset_csv)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    total_word_edits = 0
    total_words = 0
    total_char_edits = 0
    total_chars = 0

    LOGGER.info("Evaluating %s samples on device=%s", len(rows), device)
    for idx, (audio_path, ref_text) in enumerate(rows, start=1):
        hyp_text = transcribe_file(processor, model, audio_path, device, args.num_beams)

        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(hyp_text)

        w_edits, w_total, _ = word_error_rate(ref_norm, hyp_norm)
        c_edits, c_total, _ = char_error_rate(ref_norm, hyp_norm)

        total_word_edits += w_edits
        total_words += w_total
        total_char_edits += c_edits
        total_chars += c_total

        LOGGER.info(
            "[%s/%s] WER=%.3f CER=%.3f | REF: %s | HYP: %s",
            idx,
            len(rows),
            w_edits / w_total,
            c_edits / c_total,
            ref_norm,
            hyp_norm,
        )

    final_wer = total_word_edits / max(1, total_words)
    final_cer = total_char_edits / max(1, total_chars)

    LOGGER.info("===== FINAL SCORES =====")
    LOGGER.info("Samples: %s", len(rows))
    LOGGER.info("WER: %.4f (%.2f%%)", final_wer, final_wer * 100)
    LOGGER.info("CER: %.4f (%.2f%%)", final_cer, final_cer * 100)


if __name__ == "__main__":
    main()
