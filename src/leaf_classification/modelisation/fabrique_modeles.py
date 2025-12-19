from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from leaf_classification.modelisation.foret_aleatoire import ClassifieurForetAleatoire
from leaf_classification.modelisation.svm import ClassifieurSVM
from leaf_classification.modelisation.reseau_neurones import ClassifieurReseauNeurones
from leaf_classification.modelisation.gradient_boosting import ClassifieurGradientBoosting
from leaf_classification.modelisation.knn import ClassifieurKNN
from leaf_classification.modelisation.regression_logistique import ClassifieurRegressionLogistique

@dataclass
class FabriqueModeles:
    """Factory pour instancier des modèles à partir d'un nom."""

    registre_modeles: Dict[str, Type[ClassifieurBase]] = field(
        default_factory=lambda: {
    "random_forest": ClassifieurForetAleatoire,
    "svm": ClassifieurSVM,
    "mlp": ClassifieurReseauNeurones,
    "gradient_boosting": ClassifieurGradientBoosting,
    "knn": ClassifieurKNN,
    "logistic_regression": ClassifieurRegressionLogistique,
    "mlp_tuned": ClassifieurReseauNeurones,
        }
    )

    def creer_modele(self, nom: str, parametres: Dict[str, Any] | None = None) -> ClassifieurBase:
        if nom not in self.registre_modeles:
            raise ValueError(f"Modèle inconnu: {nom}. Disponibles: {self.obtenir_modeles_disponibles()}")

        cls = self.registre_modeles[nom]
        parametres = parametres or {}
        return cls(hyperparametres=parametres)

    def obtenir_modeles_disponibles(self) -> List[str]:
        return sorted(self.registre_modeles.keys())
