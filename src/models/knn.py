from sklearn.neighbors import KNeighborsClassifier

from src.models.base_classifier import BaseClassifier


class KNNClassifierModel(BaseClassifier):
    """K-Nearest Neighbors classifier wrapper"""

    def __init__(self, random_state: int = 42):
        super().__init__(random_state)

        self.param_grid = {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform"],
            "metric": ["euclidean"],
        }

    def build_model(self):
        return KNeighborsClassifier()
