import argparse
import csv
import logging
import os
import random
from typing import Dict, List

import librosa
import soundfile as sf
from audiomentations import (
    AddGaussianNoise,
    Compose,
    Mp3Compression,
    PitchShift,
    Shift,
    TimeStretch,
)


DEFAULT_INPUT_CSV = "dataset/processed/data_clean.csv"
DEFAULT_OUTPUT_CSV = "dataset/processed/data_augmented.csv"
DEFAULT_OUTPUT_AUDIO_DIR = "dataset/audio/augmented"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("augment_dataset")


def parse_args():
    parser = argparse.ArgumentParser(description="Virtual dataset expansion with audiomentations")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="CSV with columns audio,text")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output merged CSV")
    parser.add_argument("--output-audio-dir", default=DEFAULT_OUTPUT_AUDIO_DIR, help="Output directory for augmented audio")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument(
        "--modes",
        default="noise,speed,pitch",
        help="Comma-separated augmentation modes: noise,speed,pitch,shift,compression",
    )
    parser.add_argument("--copies-per-mode", type=int, default=1, help="How many variants per mode for each sample")
    parser.add_argument("--include-original", action="store_true", help="Include original rows in output CSV")
    parser.add_argument("--max-rows", type=int, default=0, help="Process only first N rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def build_transforms() -> Dict[str, Compose]:
    return {
        "noise": Compose(
            [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=1.0)]
        ),
        "speed": Compose(
            [TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=False, p=1.0)]
        ),
        "pitch": Compose([PitchShift(min_semitones=-2.0, max_semitones=2.0, p=1.0)]),
        "shift": Compose([Shift(min_shift=-0.1, max_shift=0.1, rollover=False, p=1.0)]),
        "compression": Compose([Mp3Compression(min_bitrate=48, max_bitrate=96, p=1.0)]),
    }


def to_absolute_audio_path(audio_value: str) -> str:
    path = audio_value.strip().replace("\\", "/")
    if path.startswith("dataset/"):
        return path
    return os.path.join("dataset", path).replace("\\", "/")


def to_dataset_relative(path: str) -> str:
    norm = path.replace("\\", "/")
    if norm.startswith("dataset/"):
        return norm[len("dataset/") :]
    return norm


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = parse_args()
    random.seed(args.seed)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    transforms = build_transforms()
    requested_modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in requested_modes if m not in transforms]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Supported: {list(transforms.keys())}")

    ensure_parent(args.output_csv)
    os.makedirs(args.output_audio_dir, exist_ok=True)

    with open(args.input_csv, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    out_rows: List[Dict[str, str]] = []
    created = 0

    if args.include_original:
        for row in rows:
            audio_val = (row.get("audio") or row.get("\ufeffaudio") or "").strip()
            text_val = (row.get("text") or "").strip()
            if audio_val and text_val:
                out_rows.append({"audio": audio_val, "text": text_val})

    for idx, row in enumerate(rows, start=1):
        audio_val = (row.get("audio") or row.get("\ufeffaudio") or "").strip()
        text_val = (row.get("text") or "").strip()
        if not audio_val or not text_val:
            continue

        src_audio = to_absolute_audio_path(audio_val)
        if not os.path.exists(src_audio):
            LOGGER.warning("[%s] Missing audio, skipped: %s", idx, src_audio)
            continue

        samples, sr = librosa.load(src_audio, sr=args.sample_rate, mono=True)
        base = os.path.splitext(os.path.basename(src_audio))[0]

        local_created = 0
        for mode in requested_modes:
            aug = transforms[mode]
            mode_dir = os.path.join(args.output_audio_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)

            for copy_i in range(1, args.copies_per_mode + 1):
                aug_samples = aug(samples=samples, sample_rate=args.sample_rate)
                out_name = f"{base}__aug_{mode}_{copy_i:02d}.wav"
                out_audio = os.path.join(mode_dir, out_name)
                sf.write(out_audio, aug_samples, args.sample_rate)

                out_rows.append({"audio": to_dataset_relative(out_audio), "text": text_val})
                local_created += 1
                created += 1

        LOGGER.info("[%s/%s] %s -> %s augmented files", idx, len(rows), src_audio, local_created)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(out_rows)

    LOGGER.info("=== Augmentation complete ===")
    LOGGER.info("Generated augmented rows: %s", created)
    LOGGER.info("Total rows in output CSV: %s", len(out_rows))
    LOGGER.info("Output CSV: %s", args.output_csv)
    LOGGER.info("Output audio root: %s", args.output_audio_dir)


if __name__ == "__main__":
    main()
