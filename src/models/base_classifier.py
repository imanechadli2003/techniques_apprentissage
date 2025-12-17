from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    """Abstract base class for all classifiers"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.param_grid = {}

    @abstractmethod
    def build_model(self):
        pass

    def get_model(self):
        if self.model is None:
            self.model = self.build_model()
        return self.model

    def get_param_grid(self):
        return self.param_grid

    def get_name(self):
        return self.__class__.__name__
