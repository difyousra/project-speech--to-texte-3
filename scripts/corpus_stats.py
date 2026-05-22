#!/usr/bin/env python3
"""Statistiques du corpus audio pour l'article."""

from __future__ import annotations

import json
import statistics
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "laststation/Audio"


def main() -> None:
    durations: list[float] = []
    failed: list[str] = []
    sr = None
    for wav in sorted(AUDIO_DIR.glob("*.wav")):
        try:
            with wave.open(str(wav), "rb") as w:
                if sr is None:
                    sr = w.getframerate()
                durations.append(w.getnframes() / w.getframerate())
        except Exception:
            failed.append(wav.name)

    stats = {
        "fichiers_total": len(list(AUDIO_DIR.glob("*.wav"))),
        "fichiers_lisibles": len(durations),
        "fichiers_corrompus": failed,
        "sample_rate_hz": sr,
        "canaux": 1,
        "duree_moyenne_s": round(statistics.mean(durations), 2) if durations else None,
        "duree_mediane_s": round(statistics.median(durations), 2) if durations else None,
        "duree_min_s": round(min(durations), 2) if durations else None,
        "duree_max_s": round(max(durations), 2) if durations else None,
        "duree_totale_s": round(sum(durations), 1) if durations else None,
        "duree_totale_h": round(sum(durations) / 3600, 2) if durations else None,
    }
    out = ROOT / "reports/corpus_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
