from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


@dataclass
class ClassifieurBase(ABC):
    """Interface commune pour tous les classifieurs scikit-learn."""

    nom_modele: str
    hyperparametres: Dict[str, Any] = field(default_factory=dict)
    modele_sklearn: Any | None = None

    @abstractmethod
    def entrainer(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predire(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        pass

    def sauvegarder_modele(self, chemin: str | Path) -> None:
        """Sauvegarde le modèle entraîné (joblib)."""
        if self.modele_sklearn is None:
            raise RuntimeError("Aucun modèle à sauvegarder (modele_sklearn est None).")

        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.modele_sklearn, chemin)
