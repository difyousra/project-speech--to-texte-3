#!/usr/bin/env bash
# Vérifie que les services tournent et utilisent le CPU (pas de GPU requis).
set -euo pipefail

API_KEY="${API_KEY:-dev-key}"
HDR=(-H "X-API-Key: ${API_KEY}")

echo "=== Health checks ==="
curl -sf http://localhost:8000/health | python3 -m json.tool
curl -sf http://localhost:8001/health
curl -sf http://localhost:8002/health
curl -sf http://localhost:8003/health
echo "OK"

echo ""
echo "=== DEVICE dans les conteneurs (attendu: cpu) ==="
docker compose exec -T nlp python3 -c "
import os, torch
print('DEVICE env:', os.getenv('DEVICE'))
print('CUDA_VISIBLE_DEVICES:', os.getenv('CUDA_VISIBLE_DEVICES'))
print('torch.cuda.is_available():', torch.cuda.is_available())
assert not torch.cuda.is_available(), 'GPU détecté — attendu CPU only'
print('NLP: CPU only OK')
"
docker compose exec -T stt python3 -c "
import torch
assert not torch.cuda.is_available()
print('STT: CPU only OK')
"

echo ""
echo "=== Test NLP /analyze ==="
echo "(chargement des modèles NLP sur CPU — peut prendre 1 à 3 min au 1er appel)"
curl -sf --max-time 600 -X POST http://localhost:8003/api/v1/analyze \
  "${HDR[@]}" \
  -H "Content-Type: application/json" \
  -d '{"text": "coupure internet urgente depuis 12 jours"}' | python3 -m json.tool

echo ""
echo "=== Tous les tests CPU ont réussi ==="
