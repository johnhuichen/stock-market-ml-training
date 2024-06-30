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
        financials: pandas.DataFrame = FinancialsForTicker.from_file(),
    ):
        self.target_col = target_col
        self.rolling_window = rolling_window
        self.forecast_window = forecast_window

        TOTAL_ASSETS_COL = RoAColumns.TOTAL_ASSETS_COL
        ROE_COL = RoAColumns.ROE_COl

        # represent numbers in millions
        financials = financials / 1e6
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

    def get(self) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        return self.dataset_x, self.dataset_y
