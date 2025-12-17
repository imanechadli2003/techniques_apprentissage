import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """Load dataset and perform train-test split"""

    def __init__(
        self,
        data_path: str,
        target_column: str,
        test_size: float,
        random_state: int,
        drop_columns: list[str] | None = None,
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.drop_columns = drop_columns or []

    def load(self):
        """Load dataset from CSV file."""
        data = pd.read_csv(self.data_path)

        if self.target_column not in data.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset."
            )

        x = data.drop(
            columns=[self.target_column] + self.drop_columns
        )
        y = data[self.target_column]

        return x, y

    def split(self, x, y):
        """Split dataset into train and test sets."""
        return train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

    def load_and_split(self):
        """Load dataset and perform train-test split."""
        x, y = self.load()
        return self.split(x, y)
