# RECLAMATION-STT — Pipeline IA

Pipeline temps réel pour réclamations Algérie Télécom : **VAD → STT multilingue (FR/AR/Darija) → classification multi-tâches → sentiment DziriBERT**.

## Structure

- [`libs/reclamation_common/`](libs/reclamation_common/) — schémas Pydantic, prétraitement, mappings
- [`services/vad/`](services/vad/) — segmentation parole (Silero + fallback RMS)
- [`services/stt/`](services/stt/) — Whisper fine-tuné + routage langue
- [`services/nlp/`](services/nlp/) — XLM-RoBERTa multi-tâches + DziriBERT
- [`services/gateway/`](services/gateway/) — API REST + WebSocket live
- [`docs/API.md`](docs/API.md) — documentation des endpoints

## Démarrage rapide (Docker)

```bash
cp .env.example .env
# API_KEY=dev-key et DEVICE=cpu sont recommandés pour un poste sans GPU
docker compose up --build -d
bash scripts/test_cpu_stack.sh
```

**CPU uniquement** : `docker-compose.yml` force `DEVICE=cpu`, installe PyTorch CPU dans les images, et active `LOW_MEMORY=true` pour le NLP (classifieur mono-tâche léger, sans DziriBERT en mémoire).

- **Interface appel live** : http://localhost:8000/ (ou `/demo`)  
- Santé API : http://localhost:8000/health  

### Test depuis votre téléphone (même Wi‑Fi)

Le **micro est bloqué en HTTP** sur la plupart des téléphones. Utilisez **HTTPS** :

```bash
bash scripts/show_lan_url.sh
bash scripts/generate_ssl_certs.sh VOTRE_IP   # ex. 192.168.1.42
# Dans .env : LAN_PUBLIC_URL=https://VOTRE_IP:8443
docker compose up -d --build gateway
```

Sur le téléphone : ouvrez **https://VOTRE_IP:8443/** → acceptez l’avertissement certificat → appuyez sur 📞 → **Autoriser le micro**.

**Alternative sans appel live** : bouton « Enregistrer » ou « Envoyer un fichier audio » (fonctionne souvent même en HTTP).

**Pare-feu** :
```bash
sudo ufw allow 8000/tcp
sudo ufw allow 8443/tcp
```

## Développement local

```bash
pip install -r requirements/dev.txt
pip install -r requirements/nlp.txt   # pour tester le NLP

export PYTHONPATH=libs:services/nlp
export PROJECT_ROOT=$(pwd)
export DEVICE=cpu

pytest tests/ -v
```

Lancer un service :

```bash
cd services/nlp && uvicorn app.main:app --port 8003
```

## Modèles

Les poids doivent être présents sous :

- `NLP/RESULT_multiTask/MultiModel_classifier.pt`
- `NLP/local_models/dziribert_local/`
- `whisper-finetuned/{FR,AR,Darja}/checkpoint-*/`

## Capture micro (VAD legacy)

```bash
python test_vad_chunks.py --engine rms
```
