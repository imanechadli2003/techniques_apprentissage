from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from leaf_classification.modelisation.classifieur_base import ClassifieurBase


@dataclass
class ClassifieurKNN(ClassifieurBase):
    nom_modele: str = "knn"
    hyperparametres: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        params_defaut = {
            "n_neighbors": 7,
            "metric": "minkowski",
        }
        params = {**params_defaut, **self.hyperparametres}
        self.modele_sklearn = KNeighborsClassifier(**params)

    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        self.modele_sklearn.fit(X, y)

    def predire(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict_proba(X)
