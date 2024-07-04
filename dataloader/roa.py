from pathlib import Path
from typing import Optional
import pandas

from datasource.financials import FinancialsForTicker


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
        financials: Optional[pandas.DataFrame] = None,
        cache=True,
        read_financials_csv=False,
    ):
        if financials is None:
            financials = FinancialsForTicker.from_file()

        file_financials = (
            f"{target_col}-{rolling_window}-{forecast_window}-financials.csv"
        )
        file_financials = Path(__file__ + f"/../{file_financials}").resolve()
        file_x = f"{target_col}-{rolling_window}-{forecast_window}-x.csv"
        file_x = Path(__file__ + f"/../{file_x}").resolve()
        file_y = f"{target_col}-{rolling_window}-{forecast_window}-y.csv"
        file_y = Path(__file__ + f"/../{file_y}").resolve()

        if (
            cache
            and file_x.is_file()
            and file_y.is_file()
            and file_financials.is_file()
        ):
            self.dataset_x = pandas.read_csv(file_x, index_col=[0, 1])
            self.dataset_y = pandas.read_csv(file_y, index_col=[0, 1])
            self.dataset_financials = (
                pandas.read_csv(file_financials, index_col=[0, 1])
                if read_financials_csv
                else pandas.DataFrame([])
            )
        else:
            self.target_col = target_col
            self.rolling_window = rolling_window
            self.forecast_window = forecast_window

            TOTAL_ASSETS_COL = RoAColumns.TOTAL_ASSETS_COL
            ROE_COL = RoAColumns.ROE_COl

            # excluded_tickers = ["BPT.US", "CRKN.US", "CGNT.US", "MARPS.US"]
            excluded_tickers = []

            index = financials.index.isin(excluded_tickers, level=0)
            financials = financials[~index] / 1e6  # represent numbers in millions
            financials_rolling_mean = (
                financials.groupby(level=0)
                .rolling(rolling_window)
                .mean()
                .droplevel(0)
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
                .groupby(level=0)
                .rolling(forecast_window)
                .mean()
                .droplevel(0)
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
            self.dataset_financials = financials

            if cache:
                self.dataset_x.to_csv(file_x)
                self.dataset_y.to_csv(file_y)
                self.dataset_financials.to_csv(file_financials)

    def get(self) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        return self.dataset_x, self.dataset_y

    def get_financials(self) -> pandas.DataFrame:
        return self.dataset_financials
