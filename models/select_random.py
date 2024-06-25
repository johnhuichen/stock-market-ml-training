import pandas
from models.model import Model

from typing import Any


class SelectRandom(Model):
    def __init__(self, frac: float):
        assert frac > 0
        assert frac <= 1
        self.frac = frac

    def __str__(self) -> str:
        return f"Select Random Predictions Model (frac={self.frac*100:.2f}%)"

    def init_model(self) -> None:
        pass

    def fit(self, train_x: Any, train_y: Any) -> None:
        pass

    def predict(self, val_x: pandas.DataFrame) -> Any:
        selected_index = val_x.sample(frac=self.frac).index
        return val_x.index.isin(selected_index)

    def visualize(self) -> None:
        pass
