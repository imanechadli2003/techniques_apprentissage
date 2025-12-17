from dataclasses import dataclass


@dataclass
class Config:
    """Global configuration for experiments"""

    # Reproducibility
    random_state: int = 42

    # Evaluation
    cv_folds: int = 3
    scoring: str = "accuracy"
    test_size: float = 0.2

    # Paths
    data_dir: str = "data/raw"
    train_file = "train.csv.zip"

    # Execution
    n_jobs: int = -1
    verbose: int = 1

    # Preprocessing options
    use_scaler: bool = True
    use_pca: bool = False
    pca_n_components: int | None = None
