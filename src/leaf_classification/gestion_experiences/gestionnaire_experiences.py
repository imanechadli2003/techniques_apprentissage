from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from leaf_classification.gestion_donnees.leaf_data_loader import LeafDataLoader
from leaf_classification.gestion_donnees.preprocesseur_donnees import PreprocesseurDonnees
from leaf_classification.gestion_experiences.gestionnaire_configuration import (
    GestionnaireConfiguration,
)
from leaf_classification.modelisation.fabrique_modeles import FabriqueModeles
from leaf_classification.optimisation_validation.validateur_croise import ValidateurCroise


@dataclass
class GestionnaireExperiences:
    nom_experience: str
    chemin_run: Path
    configuration: Dict[str, Any]

    def executer_experience_complete(self) -> Dict[str, Any]:
        # 1) Charger données
        cfg_d = self.configuration["donnees"]
        loader = LeafDataLoader(
            chemin_train_zip=cfg_d["chemin_train_zip"],
            chemin_test_zip=cfg_d["chemin_test_zip"],
            dossier_extraction=cfg_d["dossier_extraction"],
        )
        loader.charger_donnees()
        X_df, y_ser = loader.obtenir_X_y_train()

        # 2) Préprocessing
        preproc = PreprocesseurDonnees()
        preproc.encodeur_labels.fit(y_ser.values)
        y = preproc.transformer_labels(y_ser)
        X = X_df

        # 3) Validation croisée
        cfg_v = self.configuration["validation"]
        top_k = int(self.configuration.get("top_k", 5))
        validateur = ValidateurCroise(
            n_folds=int(cfg_v.get("n_folds", 5)),
            strategie=str(cfg_v.get("strategie", "stratified")),
            random_state=int(self.configuration.get("seed", 42)),
        )

        fabrique = FabriqueModeles()
        modeles_actifs: List[str] = list(self.configuration["modeles_actifs"])

        resultats: List[Dict[str, Any]] = []

        for nom_modele in modeles_actifs:
            modele = fabrique.creer_modele(nom_modele, parametres={})
            cv = validateur.valider(modele, X, y, top_k=top_k, normaliser=True)


            ligne = {
                "modele": nom_modele,
                **cv["resume"],
                "n_folds": cv["n_folds"],
            }
            resultats.append(ligne)

        # 4) Sauvegardes
        df = pd.DataFrame(resultats).sort_values("log_loss_moyenne", ascending=True)
        chemin_csv = self.chemin_run / "comparaison.csv"
        df.to_csv(chemin_csv, index=False)

        chemin_json = self.chemin_run / "metriques.json"
        with chemin_json.open("w", encoding="utf-8") as f:
            json.dump({"resultats": resultats}, f, indent=2)

        return {"df_comparaison": df, "resultats": resultats}
