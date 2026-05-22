# Évaluation STT Darija — RECLAMATION-STT

## Jeu de données du projet

| Fichier | Contenu |
|---------|---------|
| `laststation/data_final_raw.csv` | ~1907 paires `audio_path` + `transcription` (référence) |
| `laststation/Audio/` | Fichiers `audio_XXXXXX.wav` |
| `laststation/dataset_augmented.csv` | Variantes bruit / vitesse / compression (éval optionnelle) |

Colonnes attendues pour un manifeste personnalisé :

```csv
audio_path,transcription
/path/to/file.wav,texte de référence darija
```

## Métriques à l'entraînement (checkpoint-1100)

D'après `whisper-finetuned/Darja/checkpoint-1100/trainer_state.json` :

| Métrique | Valeur (validation entraînement) |
|----------|----------------------------------|
| **WER** | **22,3 %** (meilleur pas, step 1100) |
| **CER** | **10,8 %** |

Ces chiffres sont calculés sur le **jeu de validation utilisé pendant le fine-tuning** (pas forcément identique au CSV exporté, mais même distribution).

## Lancer l'évaluation

### Prérequis

```bash
docker compose up -d stt
# Attendre STT : OK (modèle chargé ~1–2 min)
pip install jiwer httpx  # optionnel jiwer, recommandé
```

### Via l'API STT (recommandé — même chemin que la production)

```bash
python scripts/evaluate_stt.py \
  --manifest laststation/data_final_raw.csv \
  --audio-dir laststation/Audio \
  --language darija \
  --mode api \
  --stt-url http://localhost:8002 \
  --max-samples 100
```

`--max-samples 0` = évaluer **tout** le corpus (~1907 fichiers, plusieurs heures en CPU).

### En local (sans Docker)

```bash
export PROJECT_ROOT=$(pwd)
export WHISPER_DZ_PATH=whisper-finetuned/Darja/checkpoint-1100
python scripts/evaluate_stt.py --mode local --max-samples 20
```

## Sorties

- `reports/stt_eval_YYYYMMDD_HHMMSS.json` — résumé + détail par fichier
- `reports/stt_eval_YYYYMMDD_HHMMSS.csv` — référence, hypothèse, WER, CER

## Interprétation WER / CER

- **WER** (Word Error Rate) : erreurs au niveau des mots — strict pour le darija code-switché (français + arabe).
- **CER** (Character Error Rate) : plus tolérant aux variantes d'écriture (`راني` / `وراه`, `internet` / `انترنت` si non normalisé).

La convention fine-tuning (termes FR en français) est appliquée **après** STT via `transcript_convention.py` — l'évaluation compare des textes normalisés (minuscules, ponctuation retirée).

## Comparaison avec la littérature (darija algérienne)

| Travail / ressource | Type | Ordre de grandeur | Commentaire |
|---------------------|------|-------------------|-------------|
| **Votre modèle (checkpoint-1100)** | Whisper small fine-tuné | WER ~22 %, CER ~11 % | Domaine réclamations AT, code-switching |
| Whisper **base/large** zero-shot (darija) | Générique | WER souvent **40–70 %+** | Sans fine-tuning domaine |
| **ADIAT** (dialecte algérien) | Corpus / benchmarks | Variable | Référence académique darija |
| **MASC** (Multi-dialect Arabic) | Corpus multi-dialecte | Variable | Comparaisons LDC / MSA vs dialecte |
| **Common Voice Arabic** | Crowdsourced | MSA / arabe standard | Pas darija télécom |
| **faster-whisper** + fine-tune | Infra | Latence ↓ | Option prod (`STT_USE_FASTER_WHISPER`) |

Pour une comparaison **équitable** avec un article :

1. Même métrique (WER vs CER) et même normalisation.
2. Même type de parole (téléphone, réclamations vs lecture studio).
3. Même split train/val/test.

Publier dans un rapport : **WER/CER sur `data_final_raw`**, **latence moyenne par seconde d'audio**, **taille modèle** (Whisper small ~967M params).

## Mise en production — recommandations

| Critère | État actuel | Recommandation |
|---------|-------------|----------------|
| Précision domaine AT | WER ~22 % sur val | **Acceptable** pour assistant + relecture humaine |
| Latence CPU live | ~20–60 s / segment 3 s | GPU ou faster-whisper CT2 pour prod réelle |
| Stabilité | `restart` + healthcheck | OK avec `mem_limit` 6G |
| Code-switching FR | Convention fine-tuning | Documenté, ne pas « corriger » vers arabe |
| Évaluation continue | Script `evaluate_stt.py` | Lancer après chaque changement modèle |
| Seuil prod | — | WER < 30 % sur échantillon 200+ fichiers **et** revue humaine spot-check |

**Verdict** : déployable en **pilote interne** (centre d'appels avec validation agent). Pour **100 % automatique sans relecture**, viser WER < 15–18 % ou boucle de feedback humain.

## Critères d'acceptation suggérés (Algérie Télécom)

1. **WER ≤ 25 %** sur 200 fichiers de test hold-out (non vus à l'entraînement).
2. **CER ≤ 12 %** sur le même jeu.
3. Latence **< 5 s** par segment de 3 s (nécessite GPU ou faster-whisper).
4. Taux d'hallucination vide / bruit < 5 % (`transcript_validate`).

## Prochaines étapes

1. Créer un split `train/val/test` figé (JSON) et ne jamais évaluer sur le train.
2. Convertir le modèle en **CTranslate2** : `bash scripts/convert_whisper_ct2.sh`.
3. Comparer WER **avant / après** conversion et en prod GPU.
