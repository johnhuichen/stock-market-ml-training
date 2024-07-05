import pandas
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from typing import Optional

from models.model import Model


class Trainer:
    def __init__(
        self,
        dataset_x: pandas.DataFrame,
        dataset_y: pandas.DataFrame,
        split_by_index: Optional[str] = None,
        test_size: float = 0.25,
    ):
        assert len(dataset_x.columns) > 0
        assert len(dataset_y.columns) == 1

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.split_by_index = split_by_index
        self.test_size = test_size

        self.dataset = pandas.concat(
            [self.dataset_x, self.dataset_y], axis=1, join="inner"
        )
        self.y_column = dataset_y.columns.values[0]

        self.train_test_split()

    def train_test_split(self) -> None:
        if self.split_by_index:
            indices = self.dataset.index.get_level_values(self.split_by_index).unique()
            train_indices, test_indices = sklearn_train_test_split(
                indices, test_size=self.test_size
            )
            self.train_data = self.dataset[
                self.dataset.index.isin(train_indices, level=self.split_by_index)
            ]
            self.test_data = self.dataset[
                self.dataset.index.isin(test_indices, level=self.split_by_index)
            ]
        else:
            self.train_data, self.test_data = sklearn_train_test_split(
                self.dataset, test_size=self.test_size
            )

    def x_y_split(self, data) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        x = data.loc[:, data.columns != self.y_column]
        y = data.loc[:, data.columns == self.y_column]
        return x, y

    def train(self, model: Model) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        train_x, train_y = self.x_y_split(self.train_data)
        test_x, test_y = self.x_y_split(self.test_data)

        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        return predictions, test_y
