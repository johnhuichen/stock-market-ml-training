import pandas
import numpy
from sklearn.ensemble import RandomForestClassifier

from models.model import Model


class RandomForest(Model):
    def __init__(
        self, n_estimators: int = 100, min_samples_leaf: int = 5, criterion="gini"
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.init_model()

    def __str__(self):
        params = {
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
        }
        params_str = ",\n\t".join([f"{k}={v}" for k, v in params.items()])
        return f"Random Forest Model\n\t({params_str})"

    def init_model(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
        )

    def fit(self, train_x: pandas.DataFrame, train_y: pandas.DataFrame) -> None:
        self.model.fit(train_x, train_y.values.ravel())

    def predict(self, val_x: pandas.DataFrame) -> numpy.ndarray:
        return self.model.predict(val_x)
