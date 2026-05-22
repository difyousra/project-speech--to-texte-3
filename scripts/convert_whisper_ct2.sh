#!/usr/bin/env bash
# Convertit un checkpoint Hugging Face Whisper en CTranslate2 pour faster-whisper (CPU plus rapide).
# Usage: bash scripts/convert_whisper_ct2.sh whisper-finetuned/Darja/checkpoint-1100 whisper-finetuned/Darja/ct2
set -euo pipefail
SRC="${1:?chemin modèle HF}"
OUT="${2:?dossier sortie CT2}"
pip install -q 'ctranslate2>=4,<5' transformers
ct2-transformers-converter --model "$SRC" --output_dir "$OUT" --quantization int8 --force
echo "OK: modèle CT2 dans $OUT — définissez WHISPER_DZ_PATH=$OUT et STT_USE_FASTER_WHISPER=true"
