import pandas
import numpy
from sklearn.ensemble import HistGradientBoostingClassifier

from models.model import Model


class HGBClassiferModel(Model):
    def __init__(self, max_leaf_nodes=31, random_state=None):
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.init_model()

    def __str__(self):
        return str(self.model)

    def init_model(self) -> None:
        self.model = HistGradientBoostingClassifier(
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
        )

    def fit(self, train_x: pandas.DataFrame, train_y: pandas.DataFrame) -> None:
        self.model.fit(train_x, train_y.values.ravel())

    def predict(self, val_x: pandas.DataFrame) -> numpy.ndarray:
        return self.model.predict(val_x)
