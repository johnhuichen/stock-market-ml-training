from pathlib import Path
from typing import Tuple
import pandas


class RoAColumns:
    ROLLING_MEAN_SUFFIX = "RollingMean"
    ROLLING_MEAN_FUTURE_SUFFIX = "RollingMeanFuture"
    NET_INCOME_COL = "netIncome"
    TOTAL_ASSETS_COL = "totalAssets"
    ROE_COl = "returnOnEquity"

    @classmethod
    def get_col_mean(cls, col: str) -> str:
        return f"{col}{RoAColumns.ROLLING_MEAN_SUFFIX}"

    @classmethod
    def get_col_mean_future(cls, col: str) -> str:
        return f"{col}{RoAColumns.ROLLING_MEAN_FUTURE_SUFFIX}"


class RoADataLoader:
    def __init__(
        self,
        target_col=RoAColumns.NET_INCOME_COL,
        rolling_window=3,
        forecast_window=5,
    ):
        self.target_col = target_col
        self.rolling_window = rolling_window
        self.forecast_window = forecast_window

        TOTAL_ASSETS_COL = RoAColumns.TOTAL_ASSETS_COL
        ROE_COL = RoAColumns.ROE_COl

        self.financials_csv = Path(
            __file__ + "/../../data_source/financials.csv"
        ).resolve()

        financials = pandas.read_csv(self.financials_csv, index_col=[0, 1])
        # represent numbers in millions
        financials = financials / 1e6
        financials_rolling_mean = (
            financials.rolling(rolling_window)
            .mean()
            .rename(columns=RoAColumns.get_col_mean)
        )
        financials = financials.merge(
            financials_rolling_mean, left_index=True, right_index=True
        )
        dataset_x = financials[
            ~financials[RoAColumns.get_col_mean(TOTAL_ASSETS_COL)].isnull()
        ]

        dataset_y = (
            financials.loc[:, [target_col, TOTAL_ASSETS_COL]]
            .rolling(forecast_window)
            .mean()
            .rename(index=lambda y: y - forecast_window, level=1)
            .rename(columns=RoAColumns.get_col_mean_future)
        )
        dataset_y[ROE_COL] = (
            dataset_y[RoAColumns.get_col_mean_future(target_col)]
            / dataset_y[RoAColumns.get_col_mean_future(TOTAL_ASSETS_COL)]
        )
        dataset_y = dataset_y[~dataset_y[ROE_COL].isnull()]

        self.dataset_x = dataset_x.loc[dataset_x.index.isin(dataset_y.index)]
        self.dataset_y = dataset_y.loc[dataset_y.index.isin(dataset_x.index)]

    def get(self) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        return self.dataset_x, self.dataset_y