# Fine-Tuning de Whisper pour la reconnaissance automatique du darija algérien dans un contexte de réclamations télécom

> **Version :** champs calculés automatiquement (corpus, splits, validation, exemples qualitatifs).  
> **À compléter dès que possible :** WER/CER **Test** et **Baseline** (commandes en fin de document).

---

## Résumé

La reconnaissance automatique de la parole (Automatic Speech Recognition — ASR) pour les dialectes arabes reste un défi majeur en raison du manque de ressources linguistiques et de la diversité phonétique et lexicale observée dans les usages réels. Le darija algérien est particulièrement concerné, notamment dans les environnements professionnels où les interactions présentent une forte spontanéité ainsi qu’un mélange fréquent entre arabe dialectal et français. Dans les centres d’appels télécom, les utilisateurs alternent régulièrement entre les deux langues, introduisant des phénomènes de code-switching susceptibles de dégrader les performances des systèmes ASR généralistes.

Dans cette étude, nous proposons une adaptation supervisée d’un modèle Whisper appliquée à la reconnaissance du darija algérien dans un contexte de réclamations télécom. Nous construisons un corpus composé de **1 907** enregistrements audio anonymisés (dont **1 889** fichiers WAV exploitables) accompagnés de transcriptions de référence, totalisant environ **2,38 heures** de parole. Un modèle **Whisper-small** est fine-tuné sur **cinq epochs** (checkpoint **1100**) et évalué à l’aide du **WER** et du **CER**. Sur l’ensemble de **validation**, le modèle atteint **22,3 %** de WER et **10,8 %** de CER. Les résultats sur l’ensemble de **test indépendant** (287 utterances) et la comparaison avec la **baseline Whisper-small sans fine-tuning** seront reportés après exécution du protocole d’évaluation décrit en annexe. Les analyses qualitatives montrent que les erreurs principales proviennent du code-switching (termes français transcrits en arabe) et des variantes orthographiques du dialecte.

**Mots-clés :** reconnaissance automatique de la parole, darija algérien, Whisper, dialectes arabes, fine-tuning, code-switching.

---

## 1. Introduction

Les systèmes modernes de reconnaissance automatique de la parole ont connu des avancées majeures grâce aux approches d’apprentissage profond et aux modèles pré-entraînés à grande échelle. Cependant, leurs performances demeurent limitées lorsqu’ils sont appliqués à des langues ou dialectes peu représentés dans les données d’entraînement.

Le darija algérien constitue un cas particulièrement complexe. Contrairement à l’arabe standard moderne, il ne possède pas de norme orthographique universellement adoptée et présente une forte variabilité régionale. De plus, les interactions réelles contiennent fréquemment des emprunts linguistiques au français, notamment dans les domaines techniques comme les télécommunications.

Dans les centres d’appels, les clients utilisent un langage spontané comprenant interruptions, hésitations, bruit de fond ainsi qu’un mélange arabe–français. Ces caractéristiques réduisent les performances des modèles généralistes de reconnaissance vocale.

Cette étude vise à évaluer l’adaptation supervisée de Whisper-small sur un corpus de réclamations télécom en darija algérien.

**Contributions principales :**

1. construction d’un corpus spécialisé de réclamations télécom ;
2. adaptation supervisée d’un modèle Whisper-small ;
3. protocole d’évaluation quantitative sur un jeu de test indépendant (split 70/15/15) ;
4. analyse qualitative des erreurs liées au code-switching.

---

## 2. État de l’art

La reconnaissance automatique de la parole arabe a fait l’objet de nombreux travaux ces dernières années. Toutefois, une grande partie des études concerne principalement l’arabe standard moderne, alors que les dialectes régionaux demeurent relativement peu étudiés.

Les dialectes maghrébins présentent des difficultés spécifiques dues à leur diversité lexicale, phonétique et morphologique. Les systèmes entraînés sur l’arabe standard présentent généralement une dégradation importante lorsqu’ils sont appliqués à des dialectes locaux.

Les modèles auto-supervisés récents tels que wav2vec 2.0 (Baevski et al., 2020) et XLS-R (Conneau et al., 2021) ont permis des progrès importants dans les langues à faibles ressources.

Plus récemment, Whisper (Radford et al., 2022) a démontré une robustesse importante sur de nombreuses langues grâce à un apprentissage à grande échelle. Plusieurs travaux montrent que le fine-tuning de Whisper améliore significativement les performances sur les langues sous-représentées.

Cependant, peu d’études portent spécifiquement sur le darija algérien appliqué à des scénarios réels de réclamations télécom avec code-switching systématique arabe–français.

---

## 3. Corpus

### 3.1 Source des données

Le corpus utilisé provient d’enregistrements audio anonymisés correspondant à des scénarios de réclamations télécom. Les données ont été anonymisées conformément aux politiques internes de confidentialité. Les informations sensibles (noms, identifiants, numéros personnels) ont été supprimées ou masquées.

### 3.2 Description du corpus

| Caractéristique | Valeur |
|-----------------|--------|
| Nombre total d’enregistrements | **1 907** |
| Fichiers audio exploitables (WAV PCM valides) | **1 889** |
| Fichiers non lisibles (format corrompu) | **18** (audio_000351 à audio_000368) |
| Type | Réclamations télécom |
| Langue | Darija algérien + français (code-switching) |
| Format audio | WAV, PCM 16 bits, **mono 16 kHz** |
| Durée moyenne | **4,54 s** |
| Durée médiane | **4,29 s** |
| Durée minimale | **1,53 s** |
| Durée maximale | **11,49 s** |
| Durée totale (corpus lisible) | **8 576 s** (~**2 h 23 min**, soit **2,38 h**) |

Les interactions présentent une parole spontanée caractérisée par : hésitations, bruit, variations dialectales, termes techniques et phénomènes de code-switching.

*Source des statistiques : `python scripts/corpus_stats.py` → `reports/corpus_stats.json`.*

### 3.3 Répartition des données

Split **fixe** (graine aléatoire **42**, proportions 70 % / 15 % / 15 %) :

| Ensemble | Pourcentage | Nombre |
|----------|-------------|--------|
| Entraînement | 70 % | **1 334** |
| Validation | 15 % | **286** |
| Test | 15 % | **287** |
| **Total** | 100 % | **1 907** |

Fichiers générés : `laststation/splits/train.csv`, `val.csv`, `test.csv`, métadonnées `split_meta.json`.

L’ensemble de test est **totalement disjoint** des ensembles d’entraînement et de validation. Les métriques finales de l’article doivent être calculées **uniquement** sur `test.csv`.

*Commande : `python scripts/create_dataset_split.py`*

---

## 4. Méthodologie

### 4.1 Prétraitement

Avant entraînement et évaluation, les transcriptions subissent une normalisation identique pour référence et hypothèse :

- conversion en minuscules ;
- suppression de la ponctuation ;
- normalisation Unicode **NFKC** ;
- suppression des caractères spéciaux hors alphabet arabe / latin.

Implémentation : `libs/reclamation_common/stt_metrics.py` (`normalize_for_metrics`).

### 4.2 Modèle proposé

Modèle : **Whisper-small** fine-tuné sur le corpus de réclamations télécom.

| Paramètre | Valeur |
|-----------|--------|
| Architecture | Whisper-small (~967 M paramètres) |
| Nombre d’epochs | **5** |
| Pas global maximum | **1 100** |
| Checkpoint retenu | **checkpoint-1100** |
| Critère de sélection | **WER minimal sur validation** |
| Batch size (entraînement) | **4** |
| Learning rate | **1×10⁻⁵** initial, décroissance linéaire (pic journalisé ~**7,9×10⁻⁶**) |
| Early stopping | patience **5** (callbacks Hugging Face) |

Le meilleur modèle sur validation atteint **WER = 22,26 %** et **CER = 10,83 %** (`whisper-finetuned/Darja/checkpoint-1100/trainer_state.json`).

### 4.3 Protocole d’évaluation (test et baseline)

**Modèle proposé (fine-tuné)** — inférence via le service STT identique à la production :

```bash
docker compose up -d stt
python scripts/evaluate_stt.py \
  --manifest laststation/splits/test.csv \
  --audio-dir laststation/Audio \
  --language darija \
  --mode api \
  --max-samples 0
```

**Baseline (Whisper-small sans fine-tuning)** — à exécuter avec le même `test.csv` et la même normalisation ; pointer le service STT vers le modèle OpenAI `small` non adapté (ou script local équivalent).

---

## 5. Métriques d’évaluation

**Word Error Rate (WER)** — proportion d’erreurs au niveau des mots (substitutions, insertions, suppressions).

**Character Error Rate (CER)** — proportion d’erreurs au niveau des caractères (plus tolérant aux variantes orthographiques).

Calcul : bibliothèque **jiwer** après normalisation, avec repli sur implémentation interne.

---

## 6. Résultats expérimentaux

### 6.1 Résultats quantitatifs

| Système | Jeu | WER (%) | CER (%) | Statut |
|---------|-----|---------|---------|--------|
| Whisper-small **baseline** | Test (287) | **[À MESURER]** | **[À MESURER]** | Commande §4.3 |
| Whisper-small **fine-tuné** | Validation (286) | **22,3** | **10,8** | Entraînement (checkpoint-1100) |
| Whisper-small **fine-tuné** | Test (287) | **[À MESURER]** | **[À MESURER]** | `evaluate_stt.py` sur `splits/test.csv` |

**Indicateur pilote (non substituable au test complet)** — 5 premiers fichiers du corpus, API production : WER moyen **20,8 %**, CER moyen **8,1 %** (cohérent avec la validation).

**À insérer dans l’article après mesure** (exemple de formulation) :

> Sur l’ensemble de test indépendant (287 utterances), le modèle fine-tuné atteint un WER de **X,X %** et un CER de **Y,Y %**, contre **A,A %** et **B,B %** pour la baseline Whisper-small, soit une réduction relative du WER de **Z %**.

**Latence (production CPU, pilote)** : environ **60–85 s** par utterance (~4,5 s d’audio) via API Docker ; déploiement temps réel nécessite GPU ou faster-whisper/CTranslate2.

### 6.2 Analyse qualitative des erreurs

Exemples réels (échantillon pilote, checkpoint-1100) :

| Référence (extrait) | Prédiction (extrait) | Type |
|---------------------|----------------------|------|
| …يدخل **compte** ديالي | …يدخل **الكونت** ديالي | Substitution (code-switching : français → arabe) |
| …ماعطاولناش **meme pas le modem** | …ماعطاوناش **مينبال modem** | Substitution (emprunt FR mal segmenté) |
| …يومين **مين** درت اشتراك | …يومين **من** درت اشتراك | Substitution (variante dialectale / particule) |
| …وين **راه** المشكل | …وين **معو** المشكل | Substitution (expression darija) |
| فكرة **شابة** … طريقة **مليحة** | فكره **شابه** … طريقه **مليحه** | Substitution (variantes orthographiques sans تشكيل) |

**Principales sources d’erreurs :**

1. **Code-switching** (termes techniques français : *modem*, *internet*, *compte*, *configuration*) ;
2. **Variabilité orthographique** du darija (absence de norme unifiée) ;
3. **Mots techniques** et marques (*idoom*, *actel*) ;
4. **Parole spontanée** (hésitations, reprises) ;
5. **Bruit** et qualité téléphonique (18 fichiers corrompus exclus des stats durée).

---

## 7. Discussion

Les résultats de **validation** (WER ≈ 22 %) indiquent que l’adaptation supervisée améliore nettement les performances attendues d’un modèle généraliste sur ce domaine (la littérature rapporte souvent **40–70 %+** de WER pour dialectes sans adaptation, selon protocole).

**Limites :**

- Corpus de taille **modérée** (~2,4 h) : risque de sur-apprentissage domaine ;
- **18 fichiers** audio non exploitables : à exclure ou ré-encoder avant évaluation finale ;
- **Code-switching** et absence de norme orthographique : plafond de performance sans convention d’annotation stricte ;
- Comparaison avec **ADIAT**, **MASC** ou **Common Voice** : **non directe** (autres conditions d’enregistrement et tâches) ;
- **Production** : précision suffisante pour **pilote avec relecture humaine** ; automatisation totale non recommandée tant que le test hold-out et la latence ne sont pas validés.

---

## 8. Conclusion

Nous avons présenté une approche de **fine-tuning de Whisper-small** pour la reconnaissance du darija algérien en contexte de réclamations télécom, sur un corpus de **1 907** utterances (**~2,38 h**). Le checkpoint-1100 atteint **22,3 %** de WER et **10,8 %** de CER sur la validation. L’évaluation sur le **test indépendant (287 utterances)** et la **baseline** compléteront la démonstration quantitative de l’apport du fine-tuning.

**Travaux futurs :**

- augmentation du corpus et réparation des fichiers corrompus ;
- évaluation systématique baseline vs fine-tuné sur `test.csv` ;
- Whisper-medium/large si ressources GPU suffisantes ;
- analyse linguistique du code-switching ;
- comparaison avec wav2vec 2.0 / XLS-R ;
- accélération inférence (GPU, CTranslate2).

---

## Annexe A — Commandes pour finaliser les chiffres de l’article

```bash
# 1. Splits (déjà fait une fois ; refaire si le CSV change)
python scripts/create_dataset_split.py

# 2. Statistiques corpus (table §3.2)
python scripts/corpus_stats.py

# 3. Test fine-tuné (remplir WER/CER Test dans tableau §6.1)
docker compose up -d stt
python scripts/evaluate_stt.py \
  --manifest laststation/splits/test.csv \
  --audio-dir laststation/Audio \
  --mode api \
  --max-samples 0

# 4. Copier les valeurs summary.wer_mean et summary.cer_mean
#    depuis reports/stt_eval_*.json vers le tableau
```

---

## Références (à compléter selon norme revue)

1. Radford, A., et al. (2022). *Robust speech recognition via large-scale weak supervision.* arXiv:2212.04356.
2. Baevski, A., et al. (2020). *wav2vec 2.0: A framework for self-supervised learning of speech representations.* NeurIPS.
3. Conneau, A., et al. (2021). *XLS-R: Self-supervised cross-lingual speech representation learning at scale.* arXiv:2111.09296.
4. Travaux ASR **arabe dialectal** et ressources **ADIAT**, **MASC**.
5. Travaux sur **code-switching** arabe–français (Maghreb) en ASR.

---

*Dernière génération : métriques validation et corpus figés ; test/baseline en attente d’exécution `evaluate_stt.py`.*
