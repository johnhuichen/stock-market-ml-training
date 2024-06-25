import numpy
import math
import pandas
from models.model import Model

from typing import Any


class SelectTop(Model):
    def __init__(
        self,
        frac: float,
        cheatsheet: pandas.DataFrame,
        sort_by_col,
        ascending: bool = True,
    ):
        assert frac > 0
        assert frac <= 1

        self.frac = frac
        self.cheatsheet = cheatsheet
        self.sort_by_col = sort_by_col
        self.ascending = ascending

    def __str__(self) -> str:
        return f"Select Top Predictions Model {self.frac*100:.2f}%"

    def init_model(self) -> None:
        pass

    def fit(self, train_x: Any, train_y: Any) -> None:
        pass

    def predict(self, val_x: pandas.DataFrame) -> numpy.ndarray:
        predictions = self.cheatsheet.loc[val_x.index].sort_values(
            by=self.sort_by_col, ascending=self.ascending
        )
        selected_size = int(self.frac * len(val_x))
        selected_index = predictions.iloc[:selected_size].index
        return val_x.index.isin(selected_index)

    def visualize(self) -> None:
        pass
