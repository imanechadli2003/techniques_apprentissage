from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from leaf_classification.modelisation.classifieur_base import ClassifieurBase
from leaf_classification.optimisation_validation.calculateur_metriques import (
    CalculateurMetriques,
)


@dataclass
class ValidateurCroise:
    n_folds: int = 5
    strategie: str = "stratified"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.strategie != "stratified":
            raise ValueError("Seule la stratégie 'stratified' est supportée.")
        self.diviseur_cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

    def obtenir_divisions_folds(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        yield from self.diviseur_cv.split(X, y)

    def calculer_scores_moyens(self, scores_par_fold: List[Dict[str, float]]) -> Dict[str, float]:
        if not scores_par_fold:
            raise ValueError("scores_par_fold est vide.")

        metriques = scores_par_fold[0].keys()
        res: Dict[str, float] = {}

        for m in metriques:
            valeurs = np.array([s[m] for s in scores_par_fold], dtype=float)
            res[f"{m}_moyenne"] = float(valeurs.mean())
            res[f"{m}_ecart_type"] = float(valeurs.std(ddof=1)) if len(valeurs) > 1 else 0.0

        return res

    def _cloner_modele(self, modele: ClassifieurBase) -> ClassifieurBase:
        cls = modele.__class__
        return cls(hyperparametres=dict(modele.hyperparametres))

    def valider(
        self,
        modele: ClassifieurBase,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        top_k: int = 5,
        normaliser: bool = True,
    ) -> Dict[str, object]:
        # X peut être DataFrame (recommandé) ou ndarray
        X_np = X.values if isinstance(X, pd.DataFrame) else X

        calculateur = CalculateurMetriques()
        scores_par_fold: List[Dict[str, float]] = []

        for _, (idx_train, idx_val) in enumerate(self.obtenir_divisions_folds(X_np, y), start=1):
            X_train, y_train = X_np[idx_train], y[idx_train]
            X_val, y_val = X_np[idx_val], y[idx_val]

            # ✅ Scaling appris uniquement sur X_train du fold
            if normaliser:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            modele_fold = self._cloner_modele(modele)
            modele_fold.entrainer(X_train, y_train)

            y_pred = modele_fold.predire(X_val)
            y_proba = modele_fold.predire_proba(X_val)

            scores = calculateur.calculer_metriques(y_val, y_pred, y_proba, top_k=top_k)
            scores_par_fold.append(scores)

        resume = self.calculer_scores_moyens(scores_par_fold)
        return {"scores_par_fold": scores_par_fold, "resume": resume, "n_folds": self.n_folds}
