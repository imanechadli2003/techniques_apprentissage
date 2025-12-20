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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        cfg_eval = self.configuration.get("evaluation_finale", {})
        utiliser_holdout = bool(cfg_eval.get("utiliser_holdout", True))
        test_size = float(cfg_eval.get("test_size", 0.2))
        random_state = int(cfg_eval.get("random_state", self.configuration.get("seed", 42)))
        if utiliser_holdout:
             X_train_dev, X_holdout, y_train_dev, y_holdout = train_test_split(
             X_df,
             y_ser,
             test_size=test_size,
             random_state=random_state,
             stratify=y_ser,
             )
        else:
            X_train_dev, y_train_dev = X_df, y_ser
            X_holdout, y_holdout = None, None
        
        # 2) Préprocessing
        preproc = PreprocesseurDonnees()
        preproc.encodeur_labels.fit(y_train_dev.values)
        y_train = preproc.transformer_labels(y_train_dev)
        X_train = X_train_dev

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
            params_par_modele = self.configuration.get("modeles_parametres", {})
            parametres = params_par_modele.get(nom_modele, {})
            modele = fabrique.creer_modele(nom_modele, parametres=parametres)
            cv = validateur.valider(modele, X_train, y_train, top_k=top_k, normaliser=True)
            resultat_holdout = None
            if utiliser_holdout and X_holdout is not None and y_holdout is not None:
                modele_final = fabrique.creer_modele(nom_modele, parametres=parametres)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_dev.values)
                modele_final.entrainer(X_train_scaled, y_train)
                y_hold = preproc.transformer_labels(y_holdout)
                X_hold_scaled = scaler.transform(X_holdout.values)
                y_pred = modele_final.predire(X_hold_scaled)
                y_proba = modele_final.predire_proba(X_hold_scaled)
                from leaf_classification.optimisation_validation.calculateur_metriques import CalculateurMetriques
                met = CalculateurMetriques()
                resultat_holdout = met.calculer_metriques(y_hold, y_pred, y_proba, top_k=top_k)




            ligne = {
                "modele": nom_modele,
                **cv["resume"],
                "n_folds": cv["n_folds"],
            }
            if resultat_holdout is not None:
                ligne.update({f"holdout_{k}": v for k, v in resultat_holdout.items()})
            resultats.append(ligne)

        # 4) Sauvegardes
        df = pd.DataFrame(resultats).sort_values("log_loss_moyenne", ascending=True)
        chemin_csv = self.chemin_run / "comparaison.csv"
        df.to_csv(chemin_csv, index=False)

        chemin_json = self.chemin_run / "metriques.json"
        with chemin_json.open("w", encoding="utf-8") as f:
            json.dump({"resultats": resultats}, f, indent=2)

        return {"df_comparaison": df, "resultats": resultats}
