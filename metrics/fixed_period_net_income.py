import functools

import pandas


from models.model import Model
from metrics.metric import Metric
from scenarios.scenario import Scenario


@functools.total_ordering
class FixedPeriodNetIncomeMetric(Metric):
    def __init__(
        self,
        scenario: Scenario,
        model: Model,
        selected_returns: pandas.DataFrame,
        all_returns: pandas.DataFrame,
    ):
        assert len(selected_returns.columns) == 1

        self.scenario = scenario
        self.model = model

        sort_by = selected_returns.columns.values[0]
        self.selected_returns = (
            selected_returns.sort_values(by=sort_by, ascending=False)[[sort_by]] * 100
        )

        self.best_picks = self.selected_returns.iloc[:5, :]
        self.worst_picks = self.selected_returns.iloc[-5:, :]
        self.portfolio_return = self.selected_returns[sort_by].mean()

        self.all_returns = all_returns.sort_values(by=sort_by, ascending=False) * 100

        self.dataframe_formatters = {
            sort_by: "{:,.2f}%".format,
        }

    def value(self) -> float:
        return self.portfolio_return

    def __str__(self) -> str:
        divider_length = 10
        return f"""{"="*divider_length} Net Income Metrics {"="*divider_length}

Description: Measure average return of portfolio picked by the model
             Average Return = Future Net Income / Present Total Assets
Scenario: {self.scenario}
Model: {self.model}
Result: {self.portfolio_return:.2f}%
Portfolio Size: {len(self.selected_returns)}
Best Picks: {self.best_picks.to_string(formatters=self.dataframe_formatters)}
Worst Picks: {self.worst_picks.to_string(formatters=self.dataframe_formatters)}
        """

    def __eq__(self, other) -> bool:
        if not isinstance(other, FixedPeriodNetIncomeMetric):
            raise Exception(
                "NetIncomeMetric can only be compared with another NetIncomeMetric"
            )

        return other.portfolio_return == self.portfolio_return

    def __lt__(self, other) -> bool:
        if not isinstance(other, FixedPeriodNetIncomeMetric):
            raise Exception(
                "NetIncomeMetric can only be compared with another NetIncomeMetric"
            )

        return other.portfolio_return > self.portfolio_return
