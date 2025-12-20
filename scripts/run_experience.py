from __future__ import annotations
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path
racine_projet = Path(__file__).resolve().parents[1]
sys.path.append(str(racine_projet / "src"))

from leaf_classification.utils.io import charger_yaml
from leaf_classification.gestion_experiences.gestionnaire_configuration import GestionnaireConfiguration
from leaf_classification.gestion_experiences.gestionnaire_experiences import GestionnaireExperiences



def creer_dossier_run(dossier_runs: Path, nom_experience: str) -> Path:
    horodatage = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nom_dossier = f"{horodatage}__{nom_experience}"
    chemin_run = dossier_runs / nom_dossier
    chemin_run.mkdir(parents=True, exist_ok=False)
    return chemin_run


def run_experience(chemin_config: str) -> None:    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experience.yaml",
        help="Chemin vers le fichier de configuration YAML.",
    )
    args = parser.parse_args()

    config = charger_yaml(args.config)

    dossier_runs = Path(config["sorties"]["dossier_runs"])
    dossier_runs.mkdir(parents=True, exist_ok=True)

    nom_experience = config.get("nom_experience", "experience")
    chemin_run = creer_dossier_run(dossier_runs, nom_experience)

    # Copier la config utilisée dans le run (traçabilité)
    shutil.copy2(args.config, chemin_run / "config_utilisee.yaml")

    print(" Run créé :", chemin_run)
    gestion_cfg = GestionnaireConfiguration(args.config)
    config = gestion_cfg.charger_configuration()
    gestion_cfg.valider_configuration()

    orchestrateur = GestionnaireExperiences(
       nom_experience=config.get("nom_experience", "experience"),
       chemin_run=chemin_run,
       configuration=config,
     ) 

    sortie = orchestrateur.executer_experience_complete()
    print(" Comparaison sauvegardée dans :", chemin_run / "comparaison.csv")
   

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_experience(args.config)
