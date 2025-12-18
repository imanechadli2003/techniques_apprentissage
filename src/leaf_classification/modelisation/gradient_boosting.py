from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from leaf_classification.modelisation.classifieur_base import ClassifieurBase


@dataclass
class ClassifieurGradientBoosting(ClassifieurBase):
    nom_modele: str = "gradient_boosting"
    hyperparametres: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        params_defaut = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "random_state": 42,
        }
        params = {**params_defaut, **self.hyperparametres}
        self.modele_sklearn = GradientBoostingClassifier(**params)

    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        self.modele_sklearn.fit(X, y)

    def predire(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        return self.modele_sklearn.predict_proba(X)
