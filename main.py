from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Projet IFT712 — Leaf Classification"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()

    # Ajouter src/ au PYTHONPATH
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    sys.path.append(str(src_path))

    # Import tardif (après sys.path)
    from scripts.run_experience import run_experience

    run_experience(args.config)


if __name__ == "__main__":
    main()
