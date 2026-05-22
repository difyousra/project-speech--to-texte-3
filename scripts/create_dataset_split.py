#!/usr/bin/env python3
"""Crée les splits train/val/test (70/15/15) pour l'article et l'évaluation STT."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "laststation/data_final_raw.csv"
DEFAULT_AUDIO = ROOT / "laststation/Audio"
OUT_DIR = ROOT / "laststation/splits"


def resolve_name(path_in_csv: str) -> str:
    return path_in_csv.replace("\\", "/").split("/")[-1].strip()


def load_rows(manifest: Path, audio_dir: Path) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row = {k.lstrip("\ufeff").strip(): v for k, v in row.items()}
            name = resolve_name((row.get("audio_path") or "").strip())
            ref = (row.get("transcription") or "").strip()
            if not name or not ref:
                continue
            if not (audio_dir / name).is_file():
                continue
            rows.append({"audio_path": name, "transcription": ref})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["audio_path", "transcription"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audio-dir", default=str(DEFAULT_AUDIO))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    args = p.parse_args()

    rows = load_rows(Path(args.manifest), Path(args.audio_dir))
    random.seed(args.seed)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "train.csv", train)
    write_csv(OUT_DIR / "val.csv", val)
    write_csv(OUT_DIR / "test.csv", test)

    meta = {
        "seed": args.seed,
        "total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(len(test) / n, 4),
    }
    (OUT_DIR / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Fichiers écrits dans {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
