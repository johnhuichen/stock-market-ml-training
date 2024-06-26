from pathlib import Path
from typing import Tuple
import pandas


class FutureNetIncomeDataLoader:
    MEAN_SUFFIX = "Mean"
    FUTURE_MEAN_SUFFIX = "FutureMean"

    TOTAL_ASSETS_COL = "totalAssets"
    RETURN_FUTURE_COL = "returnFuture"

    def __init__(
        self,
        target_col="netIncome",
        rolling_window=5,
        forecast_window=10,
    ):
        self.target_col = target_col
        self.rolling_window = rolling_window
        self.forecast_window = forecast_window

        MEAN_SUFFIX = FutureNetIncomeDataLoader.MEAN_SUFFIX
        FUTURE_MEAN_SUFFIX = FutureNetIncomeDataLoader.FUTURE_MEAN_SUFFIX

        RETURN_FUTURE_COL = FutureNetIncomeDataLoader.RETURN_FUTURE_COL
        TOTAL_ASSETS_COL = FutureNetIncomeDataLoader.TOTAL_ASSETS_COL
        TOTAL_ASSETS_MEAN_COL = f"{TOTAL_ASSETS_COL}{MEAN_SUFFIX}"

        self.financials_csv = Path(
            __file__ + "/../../data_source/financials.csv"
        ).resolve()

        financials = pandas.read_csv(self.financials_csv, index_col=[0, 1])
        # represent numbers in millions
        financials = financials / 1e6
        financials_rolling_mean = (
            financials.rolling(rolling_window)
            .mean()
            .rename(columns=lambda col: f"{col}{MEAN_SUFFIX}")
        )
        financials = financials.merge(
            financials_rolling_mean, left_index=True, right_index=True
        )

        dataset_x = financials[~financials[TOTAL_ASSETS_MEAN_COL].isnull()]

        target_col = self.get_target_col()
        dataset_y = (
            financials.rolling(forecast_window)
            .mean()
            .rename(columns=lambda col: f"{col}{FUTURE_MEAN_SUFFIX}")
            .rename(index=lambda y: y - forecast_window, level=1)
        )
        dataset_y = dataset_y[~dataset_y[target_col].isnull()]
        dataset_y = dataset_y.loc[:, [target_col]]
        dataset_y[RETURN_FUTURE_COL] = (
            dataset_y[target_col] / dataset_x[TOTAL_ASSETS_MEAN_COL]
        )

        self.dataset_x = dataset_x.loc[dataset_x.index.isin(dataset_y.index)]
        self.dataset_y = dataset_y.loc[dataset_y.index.isin(dataset_x.index)]

    def get_target_col(self):
        return f"{self.target_col}{FutureNetIncomeDataLoader.FUTURE_MEAN_SUFFIX}"

    def get(self) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        return self.dataset_x, self.dataset_y
