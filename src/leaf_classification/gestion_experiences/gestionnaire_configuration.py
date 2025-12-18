from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from leaf_classification.utils.io import charger_yaml


@dataclass
class GestionnaireConfiguration:
    chemin_config: str
    donnees_config: Dict[str, Any] | None = None

    def charger_configuration(self) -> Dict[str, Any]:
        self.donnees_config = charger_yaml(self.chemin_config)
        return self.donnees_config

    def valider_configuration(self) -> bool:
        if self.donnees_config is None:
            self.charger_configuration()

        assert self.donnees_config is not None
        requis = ["donnees", "sorties", "validation", "modeles_actifs"]
        for cle in requis:
            if cle not in self.donnees_config:
                raise ValueError(f"Clé manquante dans la config: '{cle}'")

        return True

    def obtenir_configs_modeles(self) -> Dict[str, Dict[str, Any]]:
        # Optionnel : pour l’instant aucun hyperparam en config
        return {}

    def obtenir_grilles_hyperparametres(self) -> Dict[str, Dict[str, Any]]:
        # Plus tard (GridSearch/RandomizedSearch)
        return {}

    def obtenir_chemin_runs(self) -> Path:
        if self.donnees_config is None:
            self.charger_configuration()
        assert self.donnees_config is not None
        return Path(self.donnees_config["sorties"]["dossier_runs"])
