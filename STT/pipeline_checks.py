import argparse
import csv
import os
import shutil

REQUIRED_COLUMNS = {"audio", "text"}
DEFAULT_INPUT_CSV = "dataset/data.csv"
DEFAULT_CLEAN_CSV = "dataset/processed/data_clean.csv"
DEFAULT_TRAIN_CSV = "dataset/processed/train.csv"
DEFAULT_VALIDATION_CSV = "dataset/processed/validation.csv"
DEFAULT_TEST_CSV = "dataset/processed/test.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Validations rapides du pipeline Whisper")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--clean-csv", default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--validation-csv", default=DEFAULT_VALIDATION_CSV)
    parser.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    parser.add_argument("--check-audio", action="store_true", help="Vérifier aussi que les fichiers audio existent")
    return parser.parse_args()


def read_header(csv_path: str):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV vide ou sans en-tête: {csv_path}")
        rows = list(reader)
        return set(reader.fieldnames), rows


def validate_csv(csv_path: str, check_audio: bool = False) -> int:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    columns, rows = read_header(csv_path)
    if not REQUIRED_COLUMNS.issubset(columns):
        raise KeyError(f"Colonnes manquantes dans {csv_path}: attendu {sorted(REQUIRED_COLUMNS)}, trouvé {sorted(columns)}")

    row_count = 0
    missing_audio = 0
    for row in rows:
        row_count += 1
        if check_audio:
            audio_path = (row.get("audio") or row.get("\ufeffaudio") or "").strip()
            if audio_path and not audio_path.startswith("dataset/"):
                audio_path = os.path.join("dataset", audio_path)
            if audio_path and not os.path.exists(audio_path):
                base, ext = os.path.splitext(audio_path)
                if ext.lower() == ".wav" and os.path.exists(f"{base}.m4a"):
                    continue
                missing_audio += 1

    if check_audio and missing_audio:
        raise FileNotFoundError(f"{missing_audio} fichiers audio manquants dans {csv_path}")

    return row_count


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg introuvable dans PATH")
    if shutil.which("ffprobe") is None:
        raise EnvironmentError("ffprobe introuvable dans PATH")

    counts = {
        "input": validate_csv(args.input_csv, check_audio=args.check_audio),
        "clean": validate_csv(args.clean_csv, check_audio=args.check_audio),
        "train": validate_csv(args.train_csv, check_audio=args.check_audio),
        "validation": validate_csv(args.validation_csv, check_audio=args.check_audio),
        "test": validate_csv(args.test_csv, check_audio=args.check_audio),
    }

    print("Pipeline checks OK")
    for name, count in counts.items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
