ID2LABEL_LEVEL1 = {
    0: "Problème / Réclamation",
    1: "Inapproprié",
    2: "Question/Assistance",
    3: "Positif",
    4: "Suggestion",
}

# Apostrophe typographique comme dans les notebooks / CSV
_APOST = "\u2019"

ID2LABEL_LEVEL2 = {
    0: "Pannes et coupures générales",
    1: "Déploiement de la fibre",
    2: f"Retards d{_APOST}installation ou de mise en service",
    3: "Réclamation service",
    4: "Espace Client + My Idoom",
}

INTENT_SLUG_MAP = {
    "Problème / Réclamation": "reclamation",
    "Inapproprié": "inappropriate",
    "Question/Assistance": "question",
    "Positif": "positive",
    "Suggestion": "suggestion",
}

CATEGORY_SLUG_MAP = {
    "Pannes et coupures générales": "pannes_coupures",
    "Déploiement de la fibre": "deploiement_fibre",
    f"Retards d{_APOST}installation ou de mise en service": "retards_installation",
    "Retards d'installation ou de mise en service": "retards_installation",
    "Réclamation service": "reclamation_service",
    "Espace Client + My Idoom": "espace_client",
}

SERVICE_MAP = {
    "pannes_coupures": "internet",
    "deploiement_fibre": "fibre",
    "retards_installation": "installation",
    "reclamation_service": "service_client",
    "espace_client": "digital",
}

THEME_MAP = {
    "pannes_coupures": "connectivity",
    "deploiement_fibre": "infrastructure",
    "retards_installation": "delivery",
    "reclamation_service": "support",
    "espace_client": "self_service",
}

REQUEST_TYPE_MAP = {
    "reclamation": "complaint",
    "inappropriate": "moderation",
    "question": "inquiry",
    "positive": "feedback",
    "suggestion": "suggestion",
}

COMPLAINT_LEVEL1_INDEX = 0

# Libellés affichage — service chargé du traitement
SERVICE_LABEL_FR = {
    "internet": "Internet & connectivité",
    "fibre": "Déploiement fibre",
    "installation": "Installation / mise en service",
    "service_client": "Service client & réclamations",
    "digital": "Espace client & My Idoom",
}

# SLA indicatif de prise en charge (jours ouvrés) selon priorité
HANDLING_SLA_DAYS = {
    "critical": 1,
    "high": 2,
    "medium": 5,
    "low": 10,
}
