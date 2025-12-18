from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

from leaf_classification.modelisation.classifieur_base import ClassifieurBase


@dataclass
class ClassifieurRegressionLogistique(ClassifieurBase):
    """Régression logistique multinomiale."""

    nom_modele: str = "logistic_regression"
    hyperparametres: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        params_defaut = {
            "C": 1.0,
            "solver": "lbfgs",
        
            "max_iter": 2000,
        }
        params = {**params_defaut, **self.hyperparametres}
        self.modele_sklearn = LogisticRegression(**params)

    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        assert self.modele_sklearn is not None
        self.modele_sklearn.fit(X, y)

    def predire(self, X: np.ndarray) -> np.ndarray:
        assert self.modele_sklearn is not None
        return self.modele_sklearn.predict(X)

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.modele_sklearn is not None
        return self.modele_sklearn.predict_proba(X)
