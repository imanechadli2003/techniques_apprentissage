from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Preprocessor:
    """Data preprocessing pipeline"""

    def __init__(
        self,
        use_scaler: bool = True,
        use_pca: bool = False,
        pca_n_components: int | None = None,
    ):
        self.use_scaler = use_scaler
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        steps = []

        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))

        if self.use_pca:
            steps.append(
                ("pca", PCA(n_components=self.pca_n_components))
            )

        return Pipeline(steps)

    def fit(self, x):
        self.pipeline.fit(x)
        return self

    def transform(self, x):
        return self.pipeline.transform(x)

    def fit_transform(self, x):
        return self.pipeline.fit_transform(x)
