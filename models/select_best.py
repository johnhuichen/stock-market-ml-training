import math
import pandas
from models.model import Model

from typing import Any


class SelectBest(Model):
    PREDICTION = "prediction"
    SORT_BY = "sort_by"

    def __init__(
        self, frac: float, predictions: pandas.DataFrame, ascending: bool = True
    ):
        assert frac > 0
        assert frac <= 1
        assert predictions.columns.values[0] == SelectBest.PREDICTION
        assert predictions.columns.values[1] == SelectBest.SORT_BY

        self.frac = frac
        self.predictions = predictions
        self.ascending = ascending

    def __str__(self) -> str:
        return f"Select Top Performing Predictions (frac={self.frac*100:.2f}%)"

    def fit(self, train_x: Any, train_y: Any) -> None:
        pass

    def predict(self, val_x: pandas.DataFrame) -> Any:
        sorted_y = self.predictions.loc[val_x.index].sort_values(
            by=SelectBest.SORT_BY, ascending=self.ascending
        )
        selected_size = math.ceil(self.frac * len(val_x))
        selected_index = sorted_y.iloc[:selected_size].index
        return val_x.index.isin(selected_index).astype(int)

    def visualize(self) -> None:
        pass
