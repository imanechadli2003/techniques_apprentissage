from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreprocesseurDonnees:
    """Prétraitement: normalisation des features + encodage des labels."""

    normaliseur: StandardScaler = StandardScaler()
    encodeur_labels: LabelEncoder = LabelEncoder()
    est_ajuste: bool = False

    def ajuster(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Ajuste le normaliseur et l'encodeur de labels sur les données train."""
        self.normaliseur.fit(X.values)
        self.encodeur_labels.fit(y.values)
        self.est_ajuste = True

    def transformer(self, X: pd.DataFrame) -> np.ndarray:
        """Transforme les features avec le normaliseur."""
        if not self.est_ajuste:
            raise RuntimeError("Le préprocesseur doit être ajusté avant transformer().")
        return self.normaliseur.transform(X.values)

    def ajuster_transformer(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Ajuste puis transforme les features."""
        self.ajuster(X, y)
        return self.transformer(X)

    def transformer_labels(self, y: pd.Series) -> np.ndarray:
      """Encode les labels avec LabelEncoder."""
      if not hasattr(self.encodeur_labels, "classes_"):
        raise RuntimeError("L'encodeur de labels doit être ajusté avant transformer_labels().")
      return self.encodeur_labels.transform(y.values)

    def obtenir_noms_classes(self) -> np.ndarray:
      """Retourne la liste des classes dans l'ordre de l'encodeur."""
      if not hasattr(self.encodeur_labels, "classes_"):
        raise RuntimeError("L'encodeur de labels doit être ajusté avant obtenir_noms_classes().")
      return self.encodeur_labels.classes_

