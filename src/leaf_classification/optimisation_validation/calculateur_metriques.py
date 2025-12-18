from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, top_k_accuracy_score


@dataclass
class CalculateurMetriques:
    """Calcule un ensemble de métriques de classification."""

    liste_metriques: List[str] = field(default_factory=lambda: ["log_loss", "accuracy", "top_k_accuracy"])
    resultats: Dict[str, float] = field(default_factory=dict)

    def calculer_metriques(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        top_k: int = 5,
    ) -> Dict[str, float]:
        """Calcule les métriques demandées."""
        res: Dict[str, float] = {}

        if "accuracy" in self.liste_metriques:
            res["accuracy"] = float(accuracy_score(y_true, y_pred))

        if "log_loss" in self.liste_metriques:
            if y_proba is None:
                raise ValueError("y_proba est requis pour calculer log_loss.")
            res["log_loss"] = float(log_loss(y_true, y_proba))

        if "top_k_accuracy" in self.liste_metriques:
            if y_proba is None:
                raise ValueError("y_proba est requis pour calculer top_k_accuracy.")
            res["top_k_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=top_k))

        self.resultats = res
        return res

    def calculer_matrice_confusion(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Retourne la matrice de confusion."""
        return confusion_matrix(y_true, y_pred)

    def calculer_log_loss(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Retourne log loss."""
        return float(log_loss(y_true, y_proba))

    def calculer_top_k_accuracy(self, y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
        """Retourne top-k accuracy."""
        return float(top_k_accuracy_score(y_true, y_proba, k=k))
