from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class LeafDataLoader:
    """Charge les données du challenge Kaggle Leaf Classification."""

    chemin_train_zip: str
    chemin_test_zip: str
    dossier_extraction: str

    donnees_train: pd.DataFrame | None = None
    donnees_test: pd.DataFrame | None = None

    def _extraire_zip(self, chemin_zip: Path, dossier_sortie: Path) -> None:
        """Extrait un fichier .zip dans un dossier (si nécessaire)."""
        dossier_sortie.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(chemin_zip, "r") as zf:
            zf.extractall(dossier_sortie)

    def _trouver_csv_extrait(self, dossier: Path, prefixe: str) -> Path:
        """Trouve le fichier CSV extrait correspondant (train/test)."""
        candidats = sorted(dossier.glob(f"{prefixe}*.csv"))
        if not candidats:
            # fallback : parfois le zip contient un nom exact train.csv / test.csv
            candidats = sorted(dossier.glob("*.csv"))

        if not candidats:
            raise FileNotFoundError(f"Aucun CSV trouvé dans {dossier}")

        # Si plusieurs CSV, on prend celui qui matche le mieux le préfixe
        for c in candidats:
            if c.name.lower().startswith(prefixe):
                return c

        return candidats[0]

    def charger_donnees(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Charge et retourne les DataFrames train et test."""
        dossier_extraction = Path(self.dossier_extraction)

        train_zip = Path(self.chemin_train_zip)
        test_zip = Path(self.chemin_test_zip)

        if not train_zip.exists():
            raise FileNotFoundError(f"ZIP train introuvable: {train_zip}")
        if not test_zip.exists():
            raise FileNotFoundError(f"ZIP test introuvable: {test_zip}")

        # Extraire si pas déjà extrait
        train_csv = dossier_extraction / "train.csv"
        test_csv = dossier_extraction / "test.csv"

        if not train_csv.exists():
            self._extraire_zip(train_zip, dossier_extraction)
            # certains zips extraient dans un autre nom → on cherche
            train_csv = self._trouver_csv_extrait(dossier_extraction, "train")

        if not test_csv.exists():
            self._extraire_zip(test_zip, dossier_extraction)
            test_csv = self._trouver_csv_extrait(dossier_extraction, "test")

        self.donnees_train = pd.read_csv(train_csv)
        self.donnees_test = pd.read_csv(test_csv)

        return self.donnees_train, self.donnees_test

    def obtenir_X_y_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Retourne X_train (features) et y_train (labels)."""
        if self.donnees_train is None:
            self.charger_donnees()

        assert self.donnees_train is not None

        if "species" not in self.donnees_train.columns:
            raise ValueError("La colonne 'species' est absente du train.csv")

        y = self.donnees_train["species"]
        X = self.donnees_train.drop(columns=["species"])

        # On garde souvent l'id à part (utile pour submission), mais on peut le laisser
        return X, y

    def obtenir_X_test(self) -> pd.DataFrame:
        """Retourne X_test (features) du fichier test."""
        if self.donnees_test is None:
            self.charger_donnees()

        assert self.donnees_test is not None
        return self.donnees_test
