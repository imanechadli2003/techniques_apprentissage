class Experiment:
    """Run a machine learning experiment"""

    def __init__(self, classifier, evaluator):
        self.classifier = classifier
        self.evaluator = evaluator

    def run(self, x_train, y_train):
        results = self.evaluator.grid_search(
            classifier=self.classifier,
            x_train=x_train,
            y_train=y_train
        )

        return {
            "model_name": self.classifier.get_name(),
            "best_score": results["best_score"],
            "best_params": results["best_params"],
            "best_estimator": results["best_estimator"],
        }
