from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def sauvegarder_barplot_cv_vs_holdout(
    df: pd.DataFrame,
    metrique: str,
    dossier_sortie: Path,
) -> None:
    """
    Génère un barplot comparant la métrique CV (moyenne ± écart-type) vs Holdout.
    """
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot = df_plot.sort_values(f"{metrique}_moyenne", ascending=True)

    x = range(len(df_plot))

    # CV moyenne ± écart-type
    plt.figure(figsize=(12, 5))
    plt.bar(x, df_plot[f"{metrique}_moyenne"], yerr=df_plot[f"{metrique}_ecart_type"], capsize=4)
    plt.xticks(x, df_plot["modele"], rotation=45, ha="right")
    plt.title(f"{metrique} (Validation croisée) — moyenne ± écart-type")
    plt.ylabel(metrique)
    plt.tight_layout()
    plt.savefig(dossier_sortie / f"{metrique}_cv.png", dpi=200)
    plt.close()

    # Holdout
    holdout_col = f"holdout_{metrique}"
    if holdout_col in df_plot.columns:
        plt.figure(figsize=(12, 5))
        plt.bar(x, df_plot[holdout_col])
        plt.xticks(x, df_plot["modele"], rotation=45, ha="right")
        plt.title(f"{metrique} (Holdout test)")
        plt.ylabel(metrique)
        plt.tight_layout()
        plt.savefig(dossier_sortie / f"{metrique}_holdout.png", dpi=200)
        plt.close()

    # Figure combinée CV vs Holdout
    if holdout_col in df_plot.columns:
        plt.figure(figsize=(12, 5))
        largeur = 0.4
        x1 = [i - largeur / 2 for i in x]
        x2 = [i + largeur / 2 for i in x]

        plt.bar(x1, df_plot[f"{metrique}_moyenne"], width=largeur,
                yerr=df_plot[f"{metrique}_ecart_type"], capsize=4, label="CV")
        plt.bar(x2, df_plot[holdout_col], width=largeur, label="Holdout")

        plt.xticks(list(x), df_plot["modele"], rotation=45, ha="right")
        plt.title(f"{metrique} — CV vs Holdout")
        plt.ylabel(metrique)
        plt.legend()
        plt.tight_layout()
        plt.savefig(dossier_sortie / f"{metrique}_cv_vs_holdout.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Chemin vers comparaison.csv (run final).",
    )
    parser.add_argument(
        "--out",
        default="results/figures",
        help="Dossier de sortie des figures.",
    )
    args = parser.parse_args()

    chemin_csv = Path(args.csv)
    dossier_out = Path(args.out)

    df = pd.read_csv(chemin_csv)

    
    sauvegarder_barplot_cv_vs_holdout(df, "log_loss", dossier_out)
    sauvegarder_barplot_cv_vs_holdout(df, "accuracy", dossier_out)

    
    if "top_k_accuracy_moyenne" in df.columns:
        sauvegarder_barplot_cv_vs_holdout(df, "top_k_accuracy", dossier_out)

    print(f" Figures sauvegardées dans : {dossier_out.resolve()}")


if __name__ == "__main__":
    main()
