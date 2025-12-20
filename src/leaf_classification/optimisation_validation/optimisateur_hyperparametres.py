from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class OptimisateurHyperparametres:
    methode_optimisation: str = "randomized"  # "grid" ou "randomized"
    grille_parametres: Dict[str, Any] | None = None

    _meilleurs_parametres: Dict[str, Any] | None = None
    _meilleur_score: float | None = None

    def optimiser(
        self,
        estimateur: BaseEstimator,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = "neg_log_loss",
        n_iter: int = 25,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ) -> Tuple[Pipeline, Dict[str, Any], float]:
        """
        Optimise les hyperparamètres via CV avec une Pipeline (scaler + modèle)
        pour éviter la fuite de données.
        """
        if self.grille_parametres is None:
            raise ValueError("grille_parametres est requis pour optimiser().")

        X_np = X.values if isinstance(X, pd.DataFrame) else X

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", clone(estimateur)),
            ]
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        # Les hyperparams doivent être préfixés par "clf__"
        params = {f"clf__{k}": v for k, v in self.grille_parametres.items()}

        if self.methode_optimisation == "grid":
            recherche = GridSearchCV(
                pipeline,
                param_grid=params,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=True,
            )
        elif self.methode_optimisation == "randomized":
            recherche = RandomizedSearchCV(
                pipeline,
                param_distributions=params,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state,
                refit=True,
            )
        else:
            raise ValueError("methode_optimisation doit être 'grid' ou 'randomized'.")

        recherche.fit(X_np, y)

        self._meilleurs_parametres = recherche.best_params_
        self._meilleur_score = float(recherche.best_score_)

        return recherche.best_estimator_, recherche.best_params_, float(recherche.best_score_)

    def obtenir_meilleurs_parametres(self) -> Dict[str, Any]:
        if self._meilleurs_parametres is None:
            raise RuntimeError("Aucune optimisation n'a été exécutée.")
        return self._meilleurs_parametres

    def obtenir_meilleur_score(self) -> float:
        if self._meilleur_score is None:
            raise RuntimeError("Aucune optimisation n'a été exécutée.")
        return self._meilleur_score
