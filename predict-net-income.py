import functools
from pathlib import Path
import pandas
import numpy

from models.decision_tree import DecisionTree
from models.select_random import SelectRandom
from models.select_best import SelectBest

from models.model import Model
from metric.metric import Metric
from trainer.trainer import Trainer


@functools.total_ordering
class NetIncomeMetric(Metric):
    def __init__(
        self,
        model: Model,
        selected_returns: pandas.DataFrame,
        all_returns: pandas.DataFrame,
    ):
        assert len(selected_returns.columns) == 1

        column = selected_returns.columns.values[0]
        self.selected_returns = (
            selected_returns.sort_values(by=column, ascending=False)[[column]] * 100
        )

        self.best_picks = self.selected_returns.iloc[:5, :]
        self.worst_picks = self.selected_returns.iloc[-5:, :]
        self.portfolio_return = self.selected_returns[column].mean()

        self.all_returns = all_returns.sort_values(by=column, ascending=False) * 100

        self.model = model
        self.dataframe_formatters = {
            column: "{:,.2f}%".format,
        }

    def __str__(self) -> str:
        divider_length = 10
        return f"""{"="*divider_length} Net Income Metrics {"="*divider_length}

Description: Measure average return of portfolio picked by the model
             Average Return = Future Net Income / Present Total Assets
Model: {self.model}
Result: {self.portfolio_return:.2f}%
Portfolio Size: {len(self.selected_returns)}
Best Picks: {self.best_picks.to_string(formatters=self.dataframe_formatters)}
Worst Picks: {self.worst_picks.to_string(formatters=self.dataframe_formatters)}
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


class NetIncomePrediction:
    TOTAL_ASSETS = "totalAssets3YrMean"
    NET_INCOME = "netIncome3YrMean"
    PREDICTION = "prediction"

    def __init__(self, year_x, year_y, threshold=0.1):
        assert year_y > year_x

        self.financials_csv = Path(__file__ + "/../data/financials.csv").resolve()
        dataframe = pandas.read_csv(self.financials_csv, index_col=[0, 1])

        # only include data with valid total assets 3 year average
        # represent numbers in millions for readability
        dataset_x = dataframe[~dataframe[NetIncomePrediction.TOTAL_ASSETS].isnull()]
        dataset_x = dataset_x.xs(year_x, level=1) / 1e6

        # only include data with valid net income 3 year average
        # represent numbers in millions for readability
        all_returns = dataframe[~dataframe[NetIncomePrediction.NET_INCOME].isnull()]
        all_returns = all_returns.xs(year_y + 2, level=1) / 1e6
        all_returns[NetIncomePrediction.PREDICTION] = (
            all_returns[NetIncomePrediction.NET_INCOME]
            / dataset_x[NetIncomePrediction.TOTAL_ASSETS]
        )
        all_returns = all_returns[[NetIncomePrediction.PREDICTION]]
        self.all_returns = all_returns

        dataset_y = (all_returns > threshold).astype(int)
        self.threshold = threshold

        self.trainer = Trainer(dataset_x, dataset_y)

    def train(self, model: Model) -> NetIncomeMetric:
        predictions, val_y = self.trainer.train(model)
        selected_index = val_y.iloc[numpy.where(predictions)].index
        selected_returns = self.all_returns.loc[selected_index]

        metric = NetIncomeMetric(model, selected_returns, self.all_returns)

        return metric

    def predictions(self) -> pandas.DataFrame:
        result = self.all_returns.rename(
            columns={NetIncomePrediction.PREDICTION: SelectBest.SORT_BY}
        )
        result[SelectBest.PREDICTION] = True
        return result[[SelectBest.PREDICTION, SelectBest.SORT_BY]]


scenario = NetIncomePrediction(2016, 2021)

model = DecisionTree()
metric = scenario.train(model)
print(metric)
model.visualize()

model = SelectRandom(0.5, predictions=scenario.predictions())
metric = scenario.train(model)
print(metric)

model = SelectBest(frac=0.5, predictions=scenario.predictions(), ascending=False)
metric = scenario.train(model)
print(metric)
