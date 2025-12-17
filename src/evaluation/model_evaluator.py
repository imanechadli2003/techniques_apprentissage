from sklearn.model_selection import GridSearchCV, StratifiedKFold


class ModelEvaluator:
    """Handle model evaluation using cross-validation and hyperparameter search"""
    
    def __init__(self, cv_folds: int, scoring: str):
        self.cv_folds = cv_folds
        self.scoring = scoring

    def grid_search(self, classifier, x_train, y_train):
        model = classifier.get_model()
        param_grid = classifier.get_param_grid()

        cv_strategy = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=42
        )

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=self.scoring,
            n_jobs=-1,
            error_score=0.0
        )

        grid_search.fit(x_train, y_train)

        return {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
        }
