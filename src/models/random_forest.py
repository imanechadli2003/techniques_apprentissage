from sklearn.ensemble import RandomForestClassifier

from src.models.base_classifier import BaseClassifier


class RandomForestClassifierModel(BaseClassifier):
    """Random Forest classifier wrapper"""

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)

        self.param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }

    def build_model(self):
        return RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1
        )
