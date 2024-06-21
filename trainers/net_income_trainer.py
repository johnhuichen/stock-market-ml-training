import functools
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from typing import cast

from models.model import Model
from trainers.metric import Metric
from trainers.trainer import Trainer


@functools.total_ordering
class NetIncomeMetric(Metric):
    def __init__(self, model: Model, predictions: np.ndarray, val_data: pd.DataFrame):
        self.model = model
        self.portfolio = val_data.loc[predictions.astype(bool)]
        self.portfolio = self.portfolio.sort_values(
            by=NetIncomeTrainer.PREDICTION, ascending=False
        )[[NetIncomeTrainer.PREDICTION]]
        self.best_picks = self.portfolio.iloc[:5, :]
        self.worst_picks = self.portfolio.iloc[-5:, :]
        self.portfolio_return = self.portfolio[NetIncomeTrainer.PREDICTION].mean()

    def __str__(self) -> str:
        divider_length = 10
        return f"""{"="*divider_length} Net Income Metrics {"="*divider_length}

Model: {self.model}
Metric: Mean value of net income divided by total assets of all predicted tickers
Result: {self.portfolio_return * 100:.2f}%
Best Picks: {self.best_picks}
Worst Picks: {self.worst_picks}

{"="*divider_length} End {"="*divider_length}
        """

    def __eq__(self, other) -> bool:
        if not isinstance(other, NetIncomeMetric):
            raise Exception(
                "NetIncomeMetric can only be compared with another NetIncomeMetric"
            )

        return other.portfolio_return == self.portfolio_return

    def __lt__(self, other) -> bool:
        if not isinstance(other, NetIncomeMetric):
            raise Exception(
                "NetIncomeMetric can only be compared with another NetIncomeMetric"
            )

        return other.portfolio_return > self.portfolio_return


class NetIncomeTrainer(Trainer):
    TOTAL_ASSETS = "totalAssets3YrMean"
    NET_INCOME = "netIncome3YrMean"
    PREDICTION = "prediction"

    def __init__(self, input_year: int):
        super().__init__()

        self.input_year = input_year
        self.financials_csv = Path(__file__ + "/../../data/financials.csv").resolve()
        dataframe = pd.read_csv(self.financials_csv, index_col=[0, 1])

        # only include data with valid total assets 3 year average
        # represent numbers in millions for readability
        start_data = dataframe[~dataframe[NetIncomeTrainer.TOTAL_ASSETS].isnull()]
        start_data = start_data.xs(input_year, level=1) / 1e6

        # only include data with valid net income 3 year average
        # represent numbers in millions for readability
        future_data = dataframe[~dataframe[NetIncomeTrainer.NET_INCOME].isnull()]
        future_data = future_data.xs(input_year + 7, level=1) / 1e6
        future_data[NetIncomeTrainer.PREDICTION] = (
            future_data[NetIncomeTrainer.NET_INCOME]
            / start_data[NetIncomeTrainer.TOTAL_ASSETS]
        )
        future_data = future_data[[NetIncomeTrainer.PREDICTION]]

        self.input = pd.concat([start_data, future_data], axis=1, join="inner")

        self.shuffle_train_test()

    def split_xy(self, data) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x = data.loc[:, data.columns != NetIncomeTrainer.PREDICTION]
        y = data.loc[:, data.columns == NetIncomeTrainer.PREDICTION]
        y = (y > 10 / 100).astype(int)
        return x, y

    def shuffle_train_test(self) -> None:
        self.train_data, self.val_data = train_test_split(self.input, test_size=0.25)

    def train(self, model: Model) -> NetIncomeMetric:
        train_x, train_y = self.split_xy(self.train_data)
        val_x, _ = self.split_xy(self.val_data)

        model.fit(train_x, train_y)
        predictions = model.predict(val_x, self.val_data)

        metric = NetIncomeMetric(model, predictions, cast(pd.DataFrame, self.val_data))

        return metric
