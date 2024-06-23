import pandas
from sklearn.model_selection import train_test_split

from typing import Tuple

from models.model import Model


class Trainer:
    def __init__(self, dataset_x: pandas.DataFrame, dataset_y: pandas.DataFrame):
        assert len(dataset_x.columns) > 1
        assert len(dataset_y.columns) == 1

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.dataset = pandas.concat(
            [self.dataset_x, self.dataset_y], axis=1, join="inner"
        )
        self.y_column = dataset_y.columns.values[0]

        self.train_test_split()

    def train_test_split(self) -> None:
        self.train_data, self.val_data = train_test_split(self.dataset, test_size=0.25)

    def x_y_split(self, data) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        x = data.loc[:, data.columns != self.y_column]
        y = data.loc[:, data.columns == self.y_column]
        return x, y

    def train(self, model: Model) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        train_x, train_y = self.x_y_split(self.train_data)
        val_x, val_y = self.x_y_split(self.val_data)

        model.fit(train_x, train_y)
        predictions = model.predict(val_x)
        return predictions, val_y
