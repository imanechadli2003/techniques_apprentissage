from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from sklearn.svm import SVC

from leaf_classification.modelisation.classifieur_base import ClassifieurBase


@dataclass
class ClassifieurSVM(ClassifieurBase):
    nom_modele: str = "svm"
    hyperparametres: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        params_defaut = {
            "kernel": "rbf",
            "C": 10.0,
            "gamma": "scale",
            "probability": True,
            "random_state": 42,
        }
        params = {**params_defaut, **self.hyperparametres}
        self.modele_sklearn = SVC(**params)

    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        self.modele_sklearn.fit(X, y)

    def predire(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict_proba(X)
