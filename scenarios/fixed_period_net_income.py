from pathlib import Path
import pandas
import numpy

from models.select_top import SelectTop

from models.model import Model
from trainer.trainer import Trainer
from scenarios.scenario import Scenario
from metrics.fixed_period_net_income import FixedPeriodNetIncomeMetric


class FixedPeriodNetIncomeScenario(Scenario):
    TOTAL_ASSETS = "totalAssets3YrMean"
    NET_INCOME = "netIncome3YrMean"
    PREDICTION = "prediction"

    def __init__(self, year_x, year_y, threshold=0.1):
        assert year_y > year_x

        self.year_x = year_x
        self.year_y = year_y
        self.threshold = threshold
        self.financials_csv = Path(
            __file__ + "/../../data_source/financials.csv"
        ).resolve()
        dataframe = pandas.read_csv(self.financials_csv, index_col=[0, 1])

        # only include data with valid total assets 3 year average
        # represent numbers in millions for readability
        dataset_x = dataframe[
            ~dataframe[FixedPeriodNetIncomeScenario.TOTAL_ASSETS].isnull()
        ]
        dataset_x = dataset_x.xs(year_x, level=1) / 1e6

        # only include data with valid net income 3 year average
        # represent numbers in millions for readability
        all_returns = dataframe[
            ~dataframe[FixedPeriodNetIncomeScenario.NET_INCOME].isnull()
        ]
        all_returns = all_returns.xs(year_y + 2, level=1) / 1e6
        all_returns[FixedPeriodNetIncomeScenario.PREDICTION] = (
            all_returns[FixedPeriodNetIncomeScenario.NET_INCOME]
            / dataset_x[FixedPeriodNetIncomeScenario.TOTAL_ASSETS]
        )
        all_returns = all_returns[[FixedPeriodNetIncomeScenario.PREDICTION]]
        self.all_returns = all_returns

        dataset_y = (all_returns > threshold).astype(int)
        self.threshold = threshold

        self.trainer = Trainer(dataset_x, dataset_y)

    def __str__(self):
        params = {
            "year_x": self.year_x,
            "year_y": self.year_y,
            "threshold": self.threshold,
        }
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        return f"FixedPeriodNetIncomeScenario ({params_str})"

    def train(self, model: Model) -> FixedPeriodNetIncomeMetric:
        predictions, val_y = self.trainer.train(model)
        selected_index = val_y.iloc[numpy.where(predictions)].index
        selected_returns = self.all_returns.loc[selected_index]

        metric = FixedPeriodNetIncomeMetric(
            self, model, selected_returns, self.all_returns
        )

        return metric

    def train_test_split(self) -> None:
        self.trainer.train_test_split()

    def sorted_predictions(self) -> pandas.DataFrame:
        result = self.all_returns.rename(
            columns={FixedPeriodNetIncomeScenario.PREDICTION: SelectTop.SORT_BY}
        )
        result[SelectTop.PREDICTION] = True
        return result[[SelectTop.PREDICTION, SelectTop.SORT_BY]]
