from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from sklearn.neural_network import MLPClassifier

from leaf_classification.modelisation.classifieur_base import ClassifieurBase


@dataclass
class ClassifieurReseauNeurones(ClassifieurBase):
    nom_modele: str = "mlp"
    hyperparametres: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        params_defaut = {
            "hidden_layer_sizes": (256, 128),
            "learning_rate_init": 0.001,
            "max_iter": 2000,
            "random_state": 42,
        }
        params = {**params_defaut, **self.hyperparametres}
        self.modele_sklearn = MLPClassifier(**params)

    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        self.modele_sklearn.fit(X, y)

    def predire(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict_proba(X)
