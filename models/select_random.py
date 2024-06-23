import pandas
from models.model import Model

from typing import Any


class SelectRandom(Model):
    PREDICTION = "prediction"

    def __init__(self, frac: float, predictions: pandas.DataFrame):
        assert frac > 0
        assert frac <= 1
        assert predictions.columns.values[0] == SelectRandom.PREDICTION
        self.frac = frac
        self.predictions = predictions

    def __str__(self) -> str:
        return f"Select Random Predictions (frac={self.frac*100:.2f}%)"

    def fit(self, train_x: Any, train_y: Any) -> None:
        pass

    def predict(self, val_x: pandas.DataFrame) -> Any:
        val_y = self.predictions.loc[val_x.index]
        selected_index = val_y.sample(frac=self.frac)[SelectRandom.PREDICTION].index
        return val_x.index.isin(selected_index).astype(int)

    def visualize(self) -> None:
        pass
