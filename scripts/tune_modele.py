from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

racine_projet = Path(__file__).resolve().parents[1]
sys.path.append(str(racine_projet / "src"))

from leaf_classification.gestion_donnees.leaf_data_loader import LeafDataLoader
from leaf_classification.gestion_donnees.preprocesseur_donnees import PreprocesseurDonnees
from leaf_classification.modelisation.fabrique_modeles import FabriqueModeles
from leaf_classification.optimisation_validation.optimisateur_hyperparametres import (
    OptimisateurHyperparametres,
)


def charger_yaml(chemin: str) -> dict:
    with open(chemin, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_exp", default="configs/experience.yaml")
    parser.add_argument("--config_modeles", default="configs/modeles.yaml")
    parser.add_argument("--modele", default="mlp")  # mlp ou logistic_regression
    args = parser.parse_args()

    cfg_exp = charger_yaml(args.config_exp)
    cfg_mod = charger_yaml(args.config_modeles)

    # 1) données
    d = cfg_exp["donnees"]
    loader = LeafDataLoader(d["chemin_train_zip"], d["chemin_test_zip"], d["dossier_extraction"])
    loader.charger_donnees()
    X_df, y_ser = loader.obtenir_X_y_train()

    # 2) labels encodés (pas de scaling ici)
    preproc = PreprocesseurDonnees()
    preproc.encodeur_labels.fit(y_ser.values)
    y = preproc.transformer_labels(y_ser)

    # 3) modèle sklearn + grilles
    fab = FabriqueModeles()
    modele_wrap = fab.creer_modele(args.modele, cfg_mod["modeles"][args.modele]["parametres_defaut"])
    estimateur = modele_wrap.modele_sklearn

    grille = cfg_mod["modeles"][args.modele]["grille"]
    t = cfg_mod["tuning"]

    opt = OptimisateurHyperparametres(methode_optimisation=t["methode"], grille_parametres=grille)
    best_pipe, best_params, best_score = opt.optimiser(
        estimateur=estimateur,
        X=X_df,
        y=y,
        cv_folds=int(t["cv_folds"]),
        scoring=str(t["scoring"]),
        n_iter=int(t["n_iter"]),
        n_jobs=int(t["n_jobs"]),
        random_state=int(t["random_state"]),
        verbose=int(t["verbose"]),
    )

    print("\n Meilleur score (neg_log_loss):", best_score)
    print(" Meilleurs hyperparamètres:")
    for k, v in best_params.items():
        print("  ", k, "=", v)


if __name__ == "__main__":
    main()
