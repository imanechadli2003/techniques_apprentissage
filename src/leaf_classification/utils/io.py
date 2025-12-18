from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def charger_yaml(chemin: str | Path) -> Dict[str, Any]:
    """Charge un fichier YAML et retourne un dictionnaire."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable: {chemin}")

    with chemin.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Le YAML doit contenir un dictionnaire à la racine.")
    return data
