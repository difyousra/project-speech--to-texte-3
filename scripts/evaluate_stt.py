#!/usr/bin/env python3
"""
Évalue le modèle STT (référence vs hypothèse) sur un manifeste CSV.

Exemple :
  python scripts/evaluate_stt.py \\
    --manifest laststation/data_final_raw.csv \\
    --audio-dir laststation/Audio \\
    --language darija \\
    --max-samples 100 \\
    --mode local

  # Via API Docker :
  python scripts/evaluate_stt.py --mode api --stt-url http://localhost:8002 --max-samples 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBS = ROOT / "libs"
if str(LIBS) not in sys.path:
    sys.path.insert(0, str(LIBS))

from reclamation_common.stt_metrics import (  # noqa: E402
    char_error_rate,
    try_jiwer_metrics,
    word_error_rate,
)

TRAINING_BEST = {
    "checkpoint": "whisper-finetuned/Darja/checkpoint-1100",
    "eval_wer": 0.2226,
    "eval_cer": 0.1083,
    "note": "Métriques lues depuis trainer_state.json (jeu de validation entraînement)",
}


def resolve_audio(path_in_csv: str, audio_dir: Path) -> Path | None:
    p = Path(path_in_csv)
    if p.is_file():
        return p
    # Chemins Windows dans le CSV : C:\...\audio_000001.wav
    name = path_in_csv.replace("\\", "/").split("/")[-1].strip()
    if not name:
        return None
    candidate = audio_dir / name
    if candidate.is_file():
        return candidate
    return None


def load_manifest(manifest: Path, audio_dir: Path, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with manifest.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.lstrip("\ufeff").strip(): v for k, v in row.items()}
            audio_col = row.get("audio_path") or row.get("path") or row.get("file")
            ref_col = row.get("transcription") or row.get("reference") or row.get("text")
            if not audio_col or not ref_col:
                continue
            ap = resolve_audio(audio_col.strip(), audio_dir)
            if ap is None:
                continue
            rows.append({"audio_path": str(ap), "reference": ref_col.strip()})
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def transcribe_local(audio_path: Path, language: str) -> dict:
    os.environ.setdefault("PROJECT_ROOT", str(ROOT))
    sys.path.insert(0, str(ROOT / "services" / "stt"))
    from app.transcribe import load_audio_bytes, transcribe_audio  # type: ignore

    data = audio_path.read_bytes()
    audio = load_audio_bytes(data)
    return transcribe_audio(audio, language_hint=language, partial=False)


def transcribe_api(audio_path: Path, language: str, stt_url: str, api_key: str) -> dict:
    import httpx

    headers = {"X-API-Key": api_key} if api_key else {}
    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f.read(), "audio/wav")}
        params = {"language_hint": language, "partial": "false"}
        r = httpx.post(
            f"{stt_url.rstrip('/')}/api/v1/transcribe",
            files=files,
            params=params,
            headers=headers,
            timeout=600.0,
        )
        r.raise_for_status()
        return r.json()


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("error") is None and r.get("hypothesis")]
    wers = [r["wer"] for r in ok]
    cers = [r["cer"] for r in ok]
    if not wers:
        return {"count_ok": 0, "count_total": len(results)}
    return {
        "count_ok": len(ok),
        "count_total": len(results),
        "wer_mean": sum(wers) / len(wers),
        "wer_median": sorted(wers)[len(wers) // 2],
        "cer_mean": sum(cers) / len(cers),
        "cer_median": sorted(cers)[len(cers) // 2],
        "wer_min": min(wers),
        "wer_max": max(wers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Évaluation STT Darija (WER/CER)")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "laststation/data_final_raw.csv"),
        help="CSV : audio_path, transcription",
    )
    parser.add_argument(
        "--audio-dir",
        default=str(ROOT / "laststation/Audio"),
        help="Dossier des fichiers .wav (nom de fichier = clé)",
    )
    parser.add_argument("--language", default="darija")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = tout le manifeste")
    parser.add_argument("--mode", choices=("local", "api"), default="api")
    parser.add_argument("--stt-url", default=os.getenv("STT_SERVICE_URL", "http://localhost:8002"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", "dev-key"))
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    manifest = Path(args.manifest)
    audio_dir = Path(args.audio_dir)
    max_n = args.max_samples or None

    samples = load_manifest(manifest, audio_dir, max_n)
    if not samples:
        print("Aucun échantillon trouvé (vérifiez --manifest et --audio-dir).", file=sys.stderr)
        return 1

    print(f"Évaluation de {len(samples)} fichiers — mode={args.mode} lang={args.language}")

    results: list[dict] = []
    t0 = time.time()
    for i, sample in enumerate(samples, 1):
        ap = Path(sample["audio_path"])
        ref = sample["reference"]
        entry = {
            "audio_path": str(ap),
            "reference": ref,
            "hypothesis": "",
            "wer": None,
            "cer": None,
            "error": None,
            "duration_sec": None,
        }
        try:
            t1 = time.time()
            if args.mode == "api":
                out = transcribe_api(ap, args.language, args.stt_url, args.api_key)
            else:
                out = transcribe_local(ap, args.language)
            entry["duration_sec"] = round(time.time() - t1, 2)
            hyp = (out.get("text") or "").strip()
            entry["hypothesis"] = hyp
            j = try_jiwer_metrics(ref, hyp)
            if j:
                entry["wer"], entry["cer"] = j
            else:
                entry["wer"] = word_error_rate(ref, hyp)
                entry["cer"] = char_error_rate(ref, hyp)
        except Exception as exc:
            entry["error"] = str(exc)
        results.append(entry)
        if i % 10 == 0 or i == len(samples):
            print(f"  [{i}/{len(samples)}] dernière WER={entry.get('wer')}")

    summary = aggregate(results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "audio_dir": str(audio_dir),
        "language": args.language,
        "mode": args.mode,
        "elapsed_sec": round(time.time() - t0, 1),
        "training_baseline": TRAINING_BEST,
        "summary": summary,
        "samples": results,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"stt_eval_{stamp}.json"
    csv_path = out_dir / f"stt_eval_{stamp}.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "audio_path",
                "reference",
                "hypothesis",
                "wer",
                "cer",
                "duration_sec",
                "error",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\n=== Résumé ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nBaseline entraînement (val) : WER≈{TRAINING_BEST['eval_wer']:.2%} CER≈{TRAINING_BEST['eval_cer']:.2%}")
    if summary.get("wer_mean") is not None:
        print(
            f"Run actuel ({args.mode}) : WER moyen={summary['wer_mean']:.2%} "
            f"CER moyen={summary['cer_mean']:.2%} "
            f"({summary['count_ok']}/{summary['count_total']} OK)"
        )
    print(f"\nRapport : {json_path}")
    print(f"CSV      : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
