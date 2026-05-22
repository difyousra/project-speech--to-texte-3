# API — Pipeline Réclamation STT

## Gateway (port 8000)

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Santé agrégée gateway + services |
| GET | `/` ou `/demo` | Interface appel live (micro, STT, classification, sentiment) |
| WS | `/ws/live?session_id=` | Flux PCM int16 16 kHz → transcripts + NLP |
| POST | `/api/v1/analyze` | Proxy analyse texte |
| POST | `/api/v1/transcribe` | Proxy transcription fichier |
| POST | `/api/v1/pipeline` | Audio complet → VAD → STT → NLP |
| GET | `/api/v1/stats/dashboard` | Statistiques agrégées des réclamations (Redis) |
| POST | `/api/v1/advise` | Solution conseiller IA (DeepSeek) à partir transcription + classification |

### Conseiller DeepSeek (`POST /api/v1/advise`)

Body JSON :

```json
{
  "transcript": "ياودي عندي شهر ملي ركبولي modem …",
  "language": "darija",
  "classification": { "priority": "high", "service_label": "Installation / mise en service" },
  "sentiment": { "sentiment": "negative", "emotion": "frustration", "score": 0.75 }
}
```

Réponse : `{ "enabled": true, "solution": "…", "model": "deepseek-chat" }`

Variables : `DEEPSEEK_API_KEY`, `DEEPSEEK_ENABLED`, `DEEPSEEK_MODEL` dans `.env`.

Header optionnel : `X-API-Key` (si `API_KEY` configuré).

## VAD (port 8001)

- `POST /api/v1/segment` — fichier audio → segments WAV base64

## STT (port 8002)

- `POST /api/v1/transcribe` — fichier audio, params `language_hint`, `partial`, `session_id`

**Convention d’écriture (modèle Darija fine-tuné)** : le darija reste en arabe/latin ;
les termes français entendus à l’oral sont normalisés en français (`للانترنت` → `internet`,
`مودم` → `modem`, etc.). Voir `libs/reclamation_common/transcript_convention.py`.

## NLP (port 8003)

- `POST /api/v1/analyze` — body JSON `{ "text": "...", "session_id": "..." }`

### Exemple réponse NLP

```json
{
  "text": "الفيبر راه ميت",
  "classification": {
    "intent": "reclamation",
    "category": "deploiement_fibre",
    "service": "fibre",
    "priority": "high",
    "urgency": true,
    "theme": "infrastructure",
    "request_type": "complaint",
    "level1_label": "Problème / Réclamation",
    "level2_label": "Déploiement de la fibre",
    "confidence": { "intent": 0.91, "category": 0.84 },
    "waiting_days": null,
    "waiting_unit": null
  },
  "sentiment": {
    "sentiment": "negative",
    "emotion": "frustration",
    "score": 0.87
  }
}
```

## Démarrage

```bash
cp .env.example .env
docker compose up --build
```
