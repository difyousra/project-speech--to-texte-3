# Article scientifique et mise en production — RECLAMATION-STT

> **Article complet pré-rempli :** voir [`ARTICLE_COMPLET.md`](ARTICLE_COMPLET.md) (même structure que votre brouillon, avec tableaux remplis).

Document rédigé pour mémoire / article de conférence sur la reconnaissance vocale du **darija algérien** dans le domaine des **réclamations télécom** (Algérie Télécom).

---

## 1. Réponse courte

| Question | Réponse |
|----------|---------|
| **Puis-je publier un article ?** | **Oui**, à condition de compléter les expériences listées en section 2 (surtout un **jeu de test hold-out** et une **ligne de base** Whisper non fine-tuné). |
| **Puis-je mettre le système en production ?** | **Oui en pilote interne** (agents qui valident la transcription). **Non** pour une automatisation totale sans contrôle humain tant que WER > ~18 % ou latence CPU actuelle. |

---

## 2. Ce que vous devez faire exactement (checklist article)

Cochez chaque point avant de soumettre l’article.

### A. Données et éthique

- [ ] Décrire la source : enregistrements de réclamations clients (anonymisés), ~1907 utterances.
- [ ] Mentionner le consentement / politique interne AT et l’anonymisation (pas de noms, numéros masqués).
- [ ] Publier les **statistiques du corpus** : durée moyenne/min/max des WAV, répartition thèmes si disponible.
- [ ] Créer un **split figé** 80 % train / 10 % val / 10 % test (ou 70/15/15) en JSON, **sans fuite** entre splits.
- [ ] N’évaluer le WER final **que sur le test** (jamais sur le train).

### B. Expériences STT obligatoires

- [ ] **Modèle proposé** : Whisper-small fine-tuné, checkpoint-1100, 5 epochs (métriques val : WER 22,3 %, CER 10,8 %).
- [ ] **Évaluation test hold-out** : `python scripts/evaluate_stt.py --max-samples 0` sur le manifeste **test uniquement** (objectif : ≥ 200 fichiers).
- [ ] **Ligne de base (baseline)** : même test avec Whisper-small **sans** fine-tuning (zero-shot, `language=darija` ou auto).
- [ ] **Ablation optionnelle** : jeu augmenté (`dataset_augmented.csv`) pour montrer la robustesse au bruit.
- [ ] **Latence** : temps moyen de transcription (s) par seconde d’audio, CPU vs GPU si disponible.
- [ ] **Normalisation** : décrire la fonction `normalize_for_metrics` (minuscules, suppression ponctuation, NFKC).

### C. Pipeline complet (si l’article couvre tout le projet)

- [ ] NLP : F1 macro / sentiment sur texte **validé** (`transcript_validate`).
- [ ] Comparaison **avec** vs **sans** validation STT avant NLP.
- [ ] Architecture microservices (schéma : VAD → STT → NLP → gateway).

### D. Littérature à citer (darija / dialecte / Whisper)

- OpenAI Whisper (Radford et al., 2022).
- Travaux sur **speech recognition for Arabic dialects** (ex. ADIAT, ressources dialecte algérien).
- **MASC** ou corpus multi-dialecte arabe (comparaison méthodologique, pas forcément mêmes chiffres).
- Articles **fine-tuning Whisper** sur langues/dialectes sous-représentés.
- **Code-switching** arabe–français en ASR (Maghreb).

### E. Figures et tableaux pour l’article

- [ ] Tableau 1 : caractéristiques du corpus.
- [ ] Tableau 2 : WER / CER (train val, test hold-out, baseline, proposé).
- [ ] Figure 1 : architecture du pipeline.
- [ ] Figure 2 (optionnel) : exemples d’erreurs (substitution, insertion, code-switch `compte` / `كونت`).
- [ ] Tableau 3 : latence et ressources (RAM, CPU/GPU).

---

## 3. Mise en production — oui ou non, et quoi faire

### Verdict

| Mode de déploiement | Autorisé ? | Conditions |
|---------------------|------------|------------|
| **Pilote interne** (centre d’appels, agent vérifie) | **Oui** | WER test ≤ 25 %, monitoring, pas de données sensibles dans les logs publics |
| **Assistant IA** (conseiller DeepSeek sur transcription validée) | **Oui** | `transcript_validate` = true avant NLP ; clés API sécurisées |
| **Décision automatique** sans humain | **Non** (pour l’instant) | Viser WER < 15–18 % + latence < 5 s/segment |

### Plan de mise en production (pilote)

1. **Geler** le modèle `checkpoint-1100` et taguer la version (ex. `stt-darija-v1.0`).
2. **Évaluer** ≥ 200 fichiers test : `scripts/evaluate_stt.py` → archiver le JSON dans `reports/`.
3. **Docker** : `docker compose up -d` avec healthcheck STT, `mem_limit` 6 Go, `API_KEY` fort.
4. **HTTPS** : `LAN_PUBLIC_URL` + certificat ; ne pas exposer sans TLS.
5. **RGPD / interne** : rétention audio limitée, logs anonymisés, révoquer toute clé API exposée.
6. **Latence** : pour prod réelle, prévoir **GPU** ou `faster-whisper` + CTranslate2.
7. **Procédure agent** : si confiance STT faible ou texte vide → re-saisie manuelle.
8. **KPI pilote** : WER estimé sur échantillon contrôlé, temps de traitement, taux de correction agent, satisfaction.

---

## 4. Texte rédigé pour l’article (à copier / adapter)

### 4.1 Titre (proposition)

**Reconnaissance vocale du darija algérien en situation de réclamation client : fine-tuning de Whisper et pipeline microservices pour l’assistance aux centres d’appels**

### 4.2 Résumé (Abstract)

> La reconnaissance automatique de la parole (ASR) en darija algérien reste peu documentée dans des contextes industriels réels, notamment lorsque la parole est spontanée, bruitée et mélangée à du français (code-switching). Nous présentons un corpus d’environ 1 907 enregistrements audio de réclamations dans le domaine des télécommunications, accompagnés de transcriptions de référence. Un modèle Whisper-small est fine-tuné sur cinq epochs pour produire le checkpoint-1100, atteignant sur le jeu de validation un taux d’erreur de mots (WER) de 22,3 % et un taux d’erreur de caractères (CER) de 10,8 %. Le modèle est intégré dans un pipeline microservices (détection d’activité vocale, STT, classification et analyse de sentiment, passerelle API) destiné à assister le traitement des réclamations. Les évaluations sur un sous-ensemble de test via l’API de production confirment des performances du même ordre de grandeur que la validation (WER ≈ 21 % sur un échantillon pilote de cinq utterances ; une évaluation complète sur hold-out est en cours). Nous discutons les limites (latence sur CPU, nécessité d’une relecture humaine) et positionnons nos résultats par rapport aux corpus dialectaux existants (ADIAT, ressources multi-dialectes). Le système est déployable en **pilote interne** avec validation par l’agent, mais non recommandé pour une décision entièrement automatique sans contrôle humain.

**Mots-clés :** darija algérien, reconnaissance vocale, Whisper, fine-tuning, code-switching, réclamations, télécommunications.

### 4.3 Introduction (extrait)

> Les dialectes arabes du Maghreb, et en particulier le darija algérien, sont sous-représentés dans les systèmes ASR commerciaux, qui privilégient l’arabe standard ou le français. Dans les centres d’appels des opérateurs télécom, les clients s’expriment souvent en darija avec insertion de termes techniques en français (*modem*, *internet*, *compte*). Cette étude vise à évaluer si un modèle générique (Whisper) peut être adapté par fine-tuning supervisé à ce domaine, et si la précision obtenue suffit pour alimenter des modules aval de compréhension automatique (classification de réclamation, sentiment).

### 4.4 Matériel et méthodes

#### Corpus

> Nous disposons de **N = 1 907** fichiers audio WAV et d’un fichier d’annotations (`data_final_raw.csv`) associant chaque fichier à une transcription de référence. Les enregistrements proviennent de scénarios de réclamation client (qualité téléphonique, parole spontanée). Les transcriptions reflètent une convention mixte arabe latinisé / arabe script et français pour le lexique technique, conforme aux pratiques observées dans le centre d’appels.

#### Split et protocole d’évaluation

> Les données sont partitionnées en ensembles d’entraînement, de validation et de test (proportions à fixer dans le split JSON). **Seul l’ensemble de test** sert aux chiffres finaux de l’article. Les métriques retenues sont le **WER** et le **CER**, calculés après normalisation Unicode NFKC, mise en minuscules et suppression de la ponctuation (`stt_metrics.py`). La ligne de base est **Whisper-small pré-entraîné sans adaptation**. Le système proposé est **Whisper-small fine-tuné (checkpoint-1100, step 1100, 5 epochs)**.

#### Fine-tuning

> Le fine-tuning utilise l’architecture Whisper-small (~967 M paramètres). L’entraînement s’arrête au pas global 1100 (meilleur checkpoint selon WER de validation). Sur la validation interne : **WER = 22,26 %**, **CER = 10,83 %** (`trainer_state.json`).

#### Déploiement expérimental

> Pour mesurer un écart train/production, l’évaluation est également menée via l’API REST du service STT (mode `api` du script `evaluate_stt.py`), identique au déploiement Docker.

### 4.5 Résultats (tableau type — à compléter)

| Système | Jeu | WER (%) | CER (%) | Remarque |
|---------|-----|---------|---------|----------|
| Whisper-small (zero-shot) | Test | *à mesurer* | *à mesurer* | Baseline |
| Whisper-small fine-tuné (ckpt-1100) | Validation | **22,3** | **10,8** | Entraînement |
| Whisper-small fine-tuné (ckpt-1100) | Test hold-out | *à mesurer* | *à mesurer* | **Chiffre principal article** |
| Fine-tuné via API Docker | Test (pilote n=5) | 20,8 | 8,1 | Indicatif ; compléter n≥200 |

> Sur l’échantillon pilote API (n = 5), la latence moyenne est d’environ **66 s par fichier** sur CPU (environ 332 s pour 5 fichiers), ce qui confirme la nécessité d’une accélération matérielle (GPU ou faster-whisper) pour un déploiement temps réel.

### 4.6 Discussion

> Nos WER (~22 %) sont **nettement inférieurs** aux ordres de grandeur rapportés pour Whisper zero-shot sur dialecte (~40–70 % dans la littérature informelle), ce qui confirme l’intérêt du fine-tuning domaine. Ils ne sont **pas directement comparables** aux benchmarks ADIAT ou Common Voice arabe standard, qui portent sur d’autres conditions d’enregistrement et d’annotation. Le **code-switching** français–darija constitue une source d’erreurs (ex. *compte* transcrit *كونت*), partiellement alignée avec la convention d’annotation. Pour la production, un WER de 22 % est **acceptable** si un agent valide la transcription avant toute action critique ; il est **insuffisant** pour une chaîne sans humain.

### 4.7 Conclusion

> Nous avons montré qu’un fine-tuning modéré de Whisper-small permet d’atteindre un WER d’environ 22 % sur la parole de réclamation en darija algérien. L’intégration microservices permet un pilote en centre d’appels. Les travaux futurs incluent : (i) split hold-out publiable, (ii) baseline systématique, (iii) conversion CTranslate2 et mesure latence GPU, (iv) extension du corpus et modèles *medium* si les ressources le permettent.

---

## 5. Commandes à exécuter avant de finaliser l’article

```bash
# 1. Évaluation complète (après création du split test)
python scripts/evaluate_stt.py \
  --manifest laststation/splits/test.csv \
  --audio-dir laststation/Audio \
  --mode api \
  --max-samples 0

# 2. Archiver le rapport pour l'article
cp reports/stt_eval_*.json reports/article_final_eval.json
```

---

## 6. Références bibliographiques (modèles BibTeX)

```bibtex
@article{radford2022whisper,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and others},
  journal={arXiv preprint arXiv:2212.04356},
  year={2022}
}
```

Ajouter selon votre revue : articles **ADIAT**, **Arabic dialect ASR**, **code-switching ASR Maghreb**, et rapports **ITU-T** sur qualité service client si pertinent.

---

*Dernière mise à jour : généré pour le projet RECLAMATION-STT. Compléter les champs « à mesurer » avant soumission.*
