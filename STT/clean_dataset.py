import argparse
import csv
import logging
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("clean_dataset")

SOURCE_CSV = "dataset/data.csv"
OUTPUT_DIR = "dataset/processed"
CLEAN_CSV = os.path.join(OUTPUT_DIR, "data_clean.csv")
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VALID_CSV = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")

SUPPORTED_AUDIO_SUFFIXES = (".wav", ".m4a", ".flac", ".ogg", ".mp3")

# Corrections metier (fautes frequentes ASR / transcription) pour Algerie Telecom.
DOMAIN_CORRECTIONS = {
    "idum": "idoom",
    "myi": "my idoom",
    "my idom": "my idoom",
    "my idoome": "my idoom",
    "conexion": "connexion",
    "connexcion": "connexion",
    "conection": "connexion",
    "connecxion": "connexion",
    "ton aliter": "tonalite",
    "tonalitee": "tonalite",
    "telephon": "telephone",
    "telephonne": "telephone",
    "telecome": "telecom",
    "algerite": "algerie",
    "algeri": "algerie",
    "maline": "ma ligne",
    "ma lignee": "ma ligne",
    "debis": "debit",
    "debie": "debit",
    "debi": "debit",
    "grommie": "nombreuses",
    "resue": "recu",
    "recoie": "recois",
    "recoit pas": "recois pas",
    "reclammation": "reclamation",
    "reclamasion": "reclamation",
    "facturre": "facture",
    "facturatione": "facturation",
    "paiyement": "paiement",
    "payement": "paiement",
    "technicienne": "technicien",
    "instalation": "installation",
    "reactivere": "reactiver",
    "resilierer": "resilier",
    "modeme": "modem",
    "routeure": "routeur",
    "wifie": "wifi",
    "fiebre": "fibre",
    "fibbre": "fibre",
    "adlse": "adsl",
    "forfais": "forfait",
    "reseuxe": "reseau",
    "reseauxe": "reseau",
    "coupurre": "coupure",
    "suspandue": "suspendue",
    "superviseure": "superviseur",
    "escalassion": "escalation",
    "verificassion": "verification",
    "identifient": "identifiant",
    "mot de pass": "mot de passe",
    "mot de passee": "mot de passe",
    "mot depasse": "mot de passe",
    "espace cliant": "espace client",
    "espace clien": "espace client",
    "espace cliant": "espace client",
    "espace clienti": "espace client",
    "compte cliant": "compte client",
    "compte clien": "compte client",
    "compte blokee": "compte bloque",
    "compte bloquee": "compte bloque",
    "compte bloquer": "compte bloque",
    "blouke": "bloque",
    "debloquage": "deblocage",
    "debloquer": "debloque",
    "mot de passe oubliee": "mot de passe oublie",
    "recuperation": "recuperation",
    "verifcation": "verification",
    "verefication": "verification",
    "verfication": "verification",
    "code de verfication": "code de verification",
    "code verification": "code de verification",
    "smss": "sms",
    "facturee": "facture",
    "fature": "facture",
    "factuere": "facture",
    "factur": "facture",
    "facturasion": "facturation",
    "facturtion": "facturation",
    "montant facturee": "montant facture",
    "paiemente": "paiement",
    "paymant": "paiement",
    "paiment": "paiement",
    "paiment en ligne": "paiement en ligne",
    "payer en ligne": "paiement en ligne",
    "carte banquaire": "carte bancaire",
    "carte bancair": "carte bancaire",
    "rechargee": "recharge",
    "sold": "solde",
    "soldes": "solde",
    "connction": "connexion",
    "connextion": "connexion",
    "conixion": "connexion",
    "deconexion": "deconnexion",
    "debitte": "debit",
    "debitt": "debit",
    "debit internete": "debit internet",
    "reseaux": "reseau",
    "reseaut": "reseau",
    "reseeau": "reseau",
    "reseauu": "reseau",
    "saturee": "sature",
    "lattence": "latence",
    "pinge": "ping",
    "coupuire": "coupure",
    "couperre": "coupure",
    "pannee": "panne",
    "pane": "panne",
    "panne general": "panne generale",
    "fibr": "fibre",
    "fibre optik": "fibre optique",
    "fibre optikque": "fibre optique",
    "adsls": "adsl",
    "idoom adlse": "idoom adsl",
    "idoom fibree": "idoom fibre",
    "4 g": "4g",
    "4gee": "4g",
    "modemme": "modem",
    "modam": "modem",
    "routeurre": "routeur",
    "router": "routeur",
    "wi fi": "wifi",
    "wi-fi": "wifi",
    "wifii": "wifi",
    "repeteur": "repeteur",
    "repetiteur": "repeteur",
    "signal faiblee": "signal faible",
    "ligne telephoniquee": "ligne telephonique",
    "ligne fixe": "ligne fixe",
    "tonalitee": "tonalite",
    "tonnalite": "tonalite",
    "telephonne fixe": "telephone fixe",
    "rendez vous": "rendez-vous",
    "rendez-vouse": "rendez-vous",
    "technitien": "technicien",
    "tecnicien": "technicien",
    "intervantion": "intervention",
    "interventtion": "intervention",
    "instalatione": "installation",
    "nouvell ligne": "nouvelle ligne",
    "activasion": "activation",
    "desactivasion": "desactivation",
    "suspention": "suspension",
    "reactivasion": "reactivation",
    "resilliation": "resiliation",
    "resialiation": "resiliation",
    "reclamatione": "reclamation",
    "reclamtion": "reclamation",
    "reclamaion": "reclamation",
    "suivie": "suivi",
    "suivie dossier": "suivi dossier",
    "delais": "delai",
    "delais de traitement": "delai de traitement",
    "escalade": "escalation",
    "responsablee": "responsable",
    "superviseur techniquee": "superviseur technique",
    "service cliant": "service client",
    "service clien": "service client",
    "offree": "offre",
    "offres disponible": "offres disponibles",
    "tarifes": "tarifs",
    "abonement": "abonnement",
    "abonemente": "abonnement",
    "abonneement": "abonnement",
    "resilier abonnement": "resilier mon abonnement",
    "reactiver ligne": "reactiver ma ligne",
    "ouvrir une lignee": "ouvrir une ligne",
    "my idoomm": "my idoom",
    "idooom": "idoom",
    "algere telecome": "algerie telecom",
    "algerie telecome": "algerie telecom",
    "algerie telecome": "algerie telecom",
}

DOMAIN_REGEX_CORRECTIONS = [
    # Typos uniquement, sans toucher les formes correctes.
    (r"\bmy\s*idoomm\b", "my idoom"),
    (r"\bidooom\b", "idoom"),
    (r"\bconnexcion\b", "connexion"),
    (r"\bconnecxion\b", "connexion"),
    (r"\bconnction\b", "connexion"),
    (r"\breclamasion\b", "reclamation"),
    (r"\breclammation\b", "reclamation"),
    (r"\bverificassion\b", "verification"),
    (r"\bverfication\b", "verification"),
    (r"\btecnicien\b", "technicien"),
    (r"\btechnitien\b", "technicien"),
    (r"\bintervantion\b", "intervention"),
    (r"\binterventtion\b", "intervention"),
    (r"\bresialiation\b", "resiliation"),
    (r"\bresilliation\b", "resiliation"),
    (r"\bmodemorages\b", "redemarrages"),
    (r"\bouture\b", "routeur"),
    (r"\brotoreur\b", "routeur"),
    (r"\bwi[\s-]fi\b", "wifi"),
]

# Corrections prioritaires demandees explicitement par l'utilisateur.
USER_REQUESTED_CORRECTIONS = [
    ("bien internet", "débit internet"),
    ("de vos olue", "résolu"),
    ("ael origine", "à l'origine"),
    ("ton aliter", "tonalité"),
    ("modemorages", "redémarrages"),
    ("rotoreur", "routeur"),
    ("outure", "routeur"),
    ("messes", "My Idoom"),
    ("maline", "ma ligne"),
    ("telephon", "téléphone"),
    ("repuisure", "plusieurs"),
    ("grommie", "nombreuses"),
    ("parre", "part"),
    ("repond", "répond"),
    ("tariffe", "tarifs"),
    ("kadie", "4G"),
    ("resue", "reçois"),
    ("pas de signale", "pas de signal"),
    ("aucun signale", "aucun signal"),
    ("ete", "été"),
    ("malgre", "malgré"),
    ("probleme", "problème"),
    ("reclamation", "réclamation"),
    ("amelioration", "amélioration"),
    ("derniere", "dernière"),
    ("telephone", "téléphone"),
    ("debit", "débit"),
    ("qualite", "qualité"),
    ("numero", "numéro"),
    ("etat", "état"),
    ("acces", "accès"),
    ("abonne", "abonné"),
    ("resolu", "résolu"),
    ("annule", "annulé"),
    ("informe", "informé"),
    ("wifi", "Wi-Fi"),
    ("4g", "4G"),
    ("adsl", "ADSL"),
    ("my idoom", "My Idoom"),
    ("n a", "n'a"),
    ("n est", "n'est"),
    ("n ai", "n'ai"),
    ("j ai", "j'ai"),
    ("c est", "c'est"),
    ("il n y a", "il n'y a"),
    ("j ai oublie", "j'ai oublié"),
    ("j ai recu", "j'ai reçu"),
    ("je nai", "je n'ai"),
    ("n a pas", "n'a pas"),
    ("n est pas", "n'est pas"),
    ("n ai pas", "n'ai pas"),
    ("mot de passe oublie", "mot de passe oublié"),
    ("mot de passe oubliee", "mot de passe oubliée"),
    ("service indisponible", "service indisponible"),
    ("espace client bloque", "espace client bloqué"),
    ("compte bloque", "compte bloqué"),
    ("code de verification", "code de vérification"),
    ("reinitialiser", "réinitialiser"),
    ("recuperer", "récupérer"),
    ("probleme technique", "problème technique"),
    ("debit internet", "débit internet"),
    ("qualite de service", "qualité de service"),
    ("numero de telephone", "numéro de téléphone"),
    ("etat de ma demande", "état de ma demande"),
    ("acces au compte", "accès au compte"),
    ("abonne idoom", "abonné idoom"),
    ("dossier resolu", "dossier résolu"),
    ("rendez vous annule", "rendez-vous annulé"),
    ("merci de m informer", "merci de m'informer"),
    ("technicien est passe", "technicien est passé"),
    ("tariffe internet", "tarifs internet"),
    ("my idoom bloque", "My Idoom bloqué"),
    ("kadie internet", "4G internet"),
    ("fibre optik", "fibre optique"),
    ("fibre optikque", "fibre optique"),
    ("connexion instable", "connexion instable"),
    ("reseau sature", "réseau saturé"),
    ("ligne telephonique", "ligne téléphonique"),
    ("telephone fixe", "téléphone fixe"),
    ("a cause", "à cause"),
    ("a partir", "à partir"),
    ("a jour", "à jour"),
    ("a distance", "à distance"),
    ("a domicile", "à domicile"),
    ("a mon avis", "à mon avis"),
    ("a l agence", "à l'agence"),
    ("a l origine", "à l'origine"),
]

# Corrections supplementaires possibles (sans retirer les regles existantes).
ADDITIONAL_DOMAIN_CORRECTIONS = {
    "aidsl": "ADSL",
    "adls": "ADSL",
    "iddoom": "idoom",
    "idoon": "idoom",
    "myidoom": "My Idoom",
    "my-idoom": "My Idoom",
    "espaceclient": "espace client",
    "compteclient": "compte client",
    "connexionn": "connexion",
    "conectione": "connexion",
    "deconexion": "deconnexion",
    "deconnexoin": "deconnexion",
    "debitt": "débit",
    "debitte": "débit",
    "latance": "latence",
    "instablee": "instable",
    "reseauxx": "réseau",
    "reseu": "réseau",
    "modamme": "modem",
    "routeuree": "routeur",
    "routeurre": "routeur",
    "wiffi": "Wi-Fi",
    "wi fi": "Wi-Fi",
    "fibreoptic": "fibre optique",
    "fibreoptique": "fibre optique",
    "telephonne": "téléphone",
    "telephonique": "téléphonique",
    "reclammation": "réclamation",
    "reclamtion": "réclamation",
    "factuere": "facture",
    "facturatione": "facturation",
    "paiment": "paiement",
    "paymant": "paiement",
    "echance": "échéance",
    "soldee": "solde",
    "technitien": "technicien",
    "intervantion": "intervention",
    "rdv": "rendez-vous",
    "rendez vous": "rendez-vous",
    "suspention": "suspension",
    "reactivasion": "réactivation",
    "resiliatione": "résiliation",
    "annuller": "annuler",
    "annuller": "annuler",
    "superviseure": "superviseur",
    "responssable": "responsable",
    "informee": "informé",
    "numeroo": "numéro",
    "accee": "accès",
    "etatt": "état",
    "qualitee": "qualité",
    "ameliorer": "améliorer",
    "amelioratione": "amélioration",
    "plusieur": "plusieurs",
    "beaucoups": "beaucoup",
}

ADDITIONAL_CONTEXT_CORRECTIONS = [
    (r"\bje\s+nai\b", "je n'ai"),
    (r"\bje\s+ne\s+peux\s+pas\s+acceder\b", "je ne peux pas accéder"),
    (r"\bn\s+y\s+a\s+pas\b", "n'y a pas"),
    (r"\bil\s+ya\b", "il y a"),
    (r"\bmot\s+de\s+pass\b", "mot de passe"),
    (r"\bmot\s+depasse\b", "mot de passe"),
    (r"\bcode\s+verfication\b", "code de vérification"),
    (r"\bservice\s+cliant\b", "service client"),
    (r"\bespace\s+cliant\b", "espace client"),
    (r"\bcompte\s+bloque\b", "compte bloqué"),
    (r"\bpas\s+de\s+connexion\b", "pas de connexion"),
    (r"\bdebit\s+internet\b", "débit internet"),
    (r"\bfibre\s+optik\b", "fibre optique"),
    (r"\bligne\s+telephonique\b", "ligne téléphonique"),
    (r"\ble\s+technicien\s+n\s+est\s+pas\s+venu\b", "le technicien n'est pas venu"),
]

# Regroupement des corrections en 4 lexiques metier.
LEXIQUE_ACCENTS = [
    ("ete", "été"),
    ("malgre", "malgré"),
    ("probleme", "problème"),
    ("reclamation", "réclamation"),
    ("amelioration", "amélioration"),
    ("derniere", "dernière"),
    ("telephone", "téléphone"),
    ("debit", "débit"),
    ("qualite", "qualité"),
    ("numero", "numéro"),
    ("etat", "état"),
    ("acces", "accès"),
    ("abonne", "abonné"),
    ("resolu", "résolu"),
    ("annule", "annulé"),
    ("informe", "informé"),
    ("a cause", "à cause"),
    ("a partir", "à partir"),
    ("a jour", "à jour"),
    ("a distance", "à distance"),
    ("a domicile", "à domicile"),
    ("a l origine", "à l'origine"),
]

LEXIQUE_CONTRACTIONS = [
    ("n a", "n'a"),
    ("n est", "n'est"),
    ("n ai", "n'ai"),
    ("j ai", "j'ai"),
    ("c est", "c'est"),
    ("il n y a", "il n'y a"),
    ("j ai oublie", "j'ai oublié"),
    ("j ai recu", "j'ai reçu"),
    ("je nai", "je n'ai"),
    ("n a pas", "n'a pas"),
    ("n est pas", "n'est pas"),
    ("n ai pas", "n'ai pas"),
]

LEXIQUE_PHONETIQUE = list(DOMAIN_CORRECTIONS.items()) + list(ADDITIONAL_DOMAIN_CORRECTIONS.items())

LEXIQUE_TELECOM = [
    ("wifi", "Wi-Fi"),
    ("4g", "4G"),
    ("adsl", "ADSL"),
    ("my idoom", "My Idoom"),
    ("bien internet", "débit internet"),
    ("ton aliter", "tonalité"),
    ("telephon", "téléphone"),
    ("maline", "ma ligne"),
    ("messes", "My Idoom"),
    ("de vos olue", "résolu"),
    ("repuisure", "plusieurs"),
    ("grommie", "nombreuses"),
    ("parre", "part"),
    ("repond", "répond"),
    ("tariffe", "tarifs"),
    ("kadie", "4G"),
    ("resue", "reçois"),
    ("outure", "routeur"),
    ("rotoreur", "routeur"),
    ("modemorages", "redémarrages"),
    ("fibre optik", "fibre optique"),
    ("fibre optikque", "fibre optique"),
]

ALL_CORRECTIONS = (
    LEXIQUE_ACCENTS +
    LEXIQUE_PHONETIQUE +
    LEXIQUE_CONTRACTIONS +
    LEXIQUE_TELECOM
)

SAFE_CONTEXT_CORRECTIONS = [
    (r"\bpas\s+de\s+signale\b", "pas de signal"),
    (r"\baucun\s+signale\b", "aucun signal"),
    (r"\bil\s+n\s+y\s+a\b", "il n'y a"),
    (r"\bn\s+a\s+pas\b", "n'a pas"),
    (r"\bn\s+est\s+pas\b", "n'est pas"),
    (r"\bn\s+ai\s+pas\b", "n'ai pas"),
    (r"\brendez\s+vous\b", "rendez-vous"),
]


@dataclass
class CleanStats:
    total_rows: int = 0
    kept_rows: int = 0
    missing_audio: int = 0
    empty_text: int = 0
    unsupported_audio_suffix: int = 0
    duplicate_audio: int = 0
    duplicate_text: int = 0
    too_short_text: int = 0
    too_long_text: int = 0
    unreadable_audio: int = 0
    too_short_audio: int = 0
    too_long_audio: int = 0
    high_special_char_ratio: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and organize the speech dataset")
    parser.add_argument("--source", default=SOURCE_CSV, help="Source CSV file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for cleaned files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--min-text-chars", type=int, default=3, help="Minimum text length to keep")
    parser.add_argument("--max-text-chars", type=int, default=400, help="Maximum text length to keep")
    parser.add_argument(
        "--max-special-char-ratio",
        type=float,
        default=0.5,
        help="Maximum ratio of non-alphanumeric non-space characters in text (0 disables)",
    )
    parser.add_argument(
        "--min-audio-duration",
        type=float,
        default=0.0,
        help="Minimum audio duration in seconds (0 disables)",
    )
    parser.add_argument(
        "--max-audio-duration",
        type=float,
        default=0.0,
        help="Maximum audio duration in seconds (0 disables)",
    )
    parser.add_argument(
        "--dedupe-text",
        action="store_true",
        help="Remove exact duplicate normalized texts",
    )
    parser.add_argument(
        "--stratify-by-length",
        action="store_true",
        help="Distribuer le split selon des buckets de longueur de texte",
    )
    parser.add_argument(
        "--length-buckets",
        type=int,
        default=5,
        help="Nombre de buckets utilises pour la stratification par longueur",
    )
    return parser.parse_args()


def correct_text(text: str) -> str:
    corrected = text.lower()

    # Normalisation des apostrophes courantes avant corrections lexicales.
    corrected = re.sub(r"\bj\s+ai\b", "j'ai", corrected)
    corrected = re.sub(r"\bd\s+([a-zàâçéèêëîïôùûüÿœæ])", r"d'\1", corrected)
    corrected = re.sub(r"\bl\s+([a-zàâçéèêëîïôùûüÿœæ])", r"l'\1", corrected)
    corrected = re.sub(r"\bc\s+est\b", "c'est", corrected)

    for pattern, replacement in SAFE_CONTEXT_CORRECTIONS:
        corrected = re.sub(pattern, replacement, corrected)

    for pattern, replacement in DOMAIN_REGEX_CORRECTIONS:
        corrected = re.sub(pattern, replacement, corrected)

    for wrong, right in sorted(ALL_CORRECTIONS, key=lambda x: len(x[0]), reverse=True):
        pattern = rf"(?<!\w){re.escape(wrong)}(?!\w)"
        corrected = re.sub(pattern, right, corrected)

    for pattern, replacement in ADDITIONAL_CONTEXT_CORRECTIONS:
        corrected = re.sub(pattern, replacement, corrected)

    # Cas demande explicitement.
    corrected = re.sub(r"\bj'ai\s+oublie\b", "j'ai oublié", corrected)
    corrected = re.sub(r"\s+", " ", corrected).strip()
    return corrected


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\xa0": " ",
        "\u200b": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = correct_text(text.strip())
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")

    return text


def resolve_audio_path(audio_value: str) -> str:
    audio_path = audio_value.strip()
    if not audio_path.startswith("dataset/"):
        audio_path = os.path.join("dataset", audio_path)
    return audio_path


def get_audio_duration(audio_path: str):
    try:
        return float(sf.info(audio_path).duration)
    except Exception:
        return None


def special_char_ratio(text: str) -> float:
    special_chars = 0
    alpha_num_chars = 0
    for ch in text:
        if ch.isalnum():
            alpha_num_chars += 1
        elif ch.isspace():
            continue
        else:
            special_chars += 1

    total = alpha_num_chars + special_chars
    if total == 0:
        return 0.0
    return special_chars / total


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "audio" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise KeyError("Le CSV doit contenir les colonnes 'audio' et 'text'.")
        return list(reader)


def write_csv(csv_path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(rows)


def clean_dataset(rows: List[Dict[str, str]], args):
    cleaned_rows = []
    stats = CleanStats(total_rows=len(rows))
    seen_audio = set()
    seen_text = set()

    for row in rows:
        audio_path = resolve_audio_path(row["audio"])
        text = normalize_text(row["text"])
        audio_ext = os.path.splitext(audio_path)[1].lower()
        text_len = len(text)

        if not text:
            stats.empty_text += 1
            continue
        if text_len < max(0, args.min_text_chars):
            stats.too_short_text += 1
            continue
        if args.max_text_chars > 0 and text_len > args.max_text_chars:
            stats.too_long_text += 1
            continue
        if args.max_special_char_ratio > 0 and special_char_ratio(text) > args.max_special_char_ratio:
            stats.high_special_char_ratio += 1
            continue
        if audio_ext and audio_ext not in SUPPORTED_AUDIO_SUFFIXES:
            stats.unsupported_audio_suffix += 1
            continue
        if not os.path.exists(audio_path):
            stats.missing_audio += 1
            continue
        if args.min_audio_duration > 0 or args.max_audio_duration > 0:
            duration = get_audio_duration(audio_path)
            if duration is None:
                stats.unreadable_audio += 1
                continue
            if args.min_audio_duration > 0 and duration < args.min_audio_duration:
                stats.too_short_audio += 1
                continue
            if args.max_audio_duration > 0 and duration > args.max_audio_duration:
                stats.too_long_audio += 1
                continue
        if audio_path in seen_audio:
            stats.duplicate_audio += 1
            continue
        if args.dedupe_text and text in seen_text:
            stats.duplicate_text += 1
            continue

        seen_audio.add(audio_path)
        seen_text.add(text)
        cleaned_rows.append({
            "audio": audio_path.replace("dataset\\", "dataset/").replace("\\", "/"),
            "text": text,
        })

    stats.kept_rows = len(cleaned_rows)
    return cleaned_rows, stats


def split_rows(rows: List[Dict[str, str]], train_ratio: float, valid_ratio: float, test_ratio: float, seed: int):
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Les ratios train/validation/test doivent totaliser 1.0")

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_rows = shuffled[:train_end]
    valid_rows = shuffled[train_end:valid_end]
    test_rows = shuffled[valid_end:]
    return train_rows, valid_rows, test_rows


def split_rows_stratified(rows: List[Dict[str, str]], train_ratio: float, valid_ratio: float, test_ratio: float, seed: int, buckets: int):
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Les ratios train/validation/test doivent totaliser 1.0")
    if buckets < 2:
        return split_rows(rows, train_ratio, valid_ratio, test_ratio, seed)

    def bucket_for(text: str) -> int:
        length = len(text.split())
        return min(buckets - 1, max(0, length // max(1, 20 // buckets)))

    grouped = {i: [] for i in range(buckets)}
    for row in rows:
        grouped[bucket_for(row["text"])].append(row)

    rng = random.Random(seed)
    train_rows: List[Dict[str, str]] = []
    valid_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for group_rows in grouped.values():
        shuffled = group_rows[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_end = int(total * train_ratio)
        valid_end = train_end + int(total * valid_ratio)
        train_rows.extend(shuffled[:train_end])
        valid_rows.extend(shuffled[train_end:valid_end])
        test_rows.extend(shuffled[valid_end:])

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    rng.shuffle(test_rows)
    return train_rows, valid_rows, test_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_rows(args.source)
    cleaned_rows, stats = clean_dataset(rows, args)
    if args.stratify_by_length:
        train_rows, valid_rows, test_rows = split_rows_stratified(
            cleaned_rows,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            args.seed,
            args.length_buckets,
        )
    else:
        train_rows, valid_rows, test_rows = split_rows(
            cleaned_rows,
            args.train_ratio,
            args.valid_ratio,
            args.test_ratio,
            args.seed,
        )

    write_csv(CLEAN_CSV, cleaned_rows)
    write_csv(TRAIN_CSV, train_rows)
    write_csv(VALID_CSV, valid_rows)
    write_csv(TEST_CSV, test_rows)

    LOGGER.info("Source rows: %s", stats.total_rows)
    LOGGER.info("Kept rows: %s", stats.kept_rows)
    LOGGER.info("Missing audio skipped: %s", stats.missing_audio)
    LOGGER.info("Empty text skipped: %s", stats.empty_text)
    LOGGER.info("Unsupported audio suffix skipped: %s", stats.unsupported_audio_suffix)
    LOGGER.info("Duplicate audio skipped: %s", stats.duplicate_audio)
    LOGGER.info("Duplicate text skipped: %s", stats.duplicate_text)
    LOGGER.info("Too short text skipped: %s", stats.too_short_text)
    LOGGER.info("Too long text skipped: %s", stats.too_long_text)
    LOGGER.info("High special-char ratio skipped: %s", stats.high_special_char_ratio)
    LOGGER.info("Unreadable audio skipped: %s", stats.unreadable_audio)
    LOGGER.info("Too short audio skipped: %s", stats.too_short_audio)
    LOGGER.info("Too long audio skipped: %s", stats.too_long_audio)
    LOGGER.info("Clean file: %s", CLEAN_CSV)
    LOGGER.info("Train split: %s (%s)", TRAIN_CSV, len(train_rows))
    LOGGER.info("Validation split: %s (%s)", VALID_CSV, len(valid_rows))
    LOGGER.info("Test split: %s (%s)", TEST_CSV, len(test_rows))


if __name__ == "__main__":
    main()
