import functools

import pandas
import numpy


from data_loader.future_net_income import FutureNetIncomeDataLoader
from models.model import Model
from metrics.metric import Metric


@functools.total_ordering
class NetIncomeMetric(Metric):
    def __init__(
        self,
        model: Model,
        predictions: pandas.DataFrame,
        val_y: pandas.DataFrame,
        future_net_incomes: pandas.DataFrame,
    ):
        self.model = model

        ROE = FutureNetIncomeDataLoader.ROE_COl
        selected_index = val_y.iloc[numpy.where(predictions)].index
        self.selected_returns = future_net_incomes.loc[selected_index, [ROE]]
        self.selected_returns = (
            self.selected_returns.sort_values(by=ROE, ascending=False)[[ROE]] * 100
        )

        self.best_picks = self.selected_returns.iloc[:5, :]
        self.worst_picks = self.selected_returns.iloc[-5:, :]
        self.portfolio_return = self.selected_returns[ROE].mean()

        self.dataframe_formatters = {
            ROE: "{:,.2f}%".format,
        }

    def value(self) -> float:
        return self.portfolio_return

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
