from sklearn.linear_model import LogisticRegression

from src.models.base_classifier import BaseClassifier


class LogisticRegressionClassifierModel(BaseClassifier):
    """Logistic Regression classifier wrapper"""

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)

        self.param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
        }

    def build_model(self):
        return LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
