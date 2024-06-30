import unittest
import pandas
import random


from dataloader.roa import RoAColumns, RoADataLoader


class TestRoADataLoader(unittest.TestCase):
    def random_data(
        self, tickers: list[tuple[str, range]]
    ) -> dict[tuple[str, int], dict[str, int]]:
        NET_INCOME_COL = RoAColumns.NET_INCOME_COL
        TOTAL_ASSETS_COL = RoAColumns.TOTAL_ASSETS_COL

        data = {}
        for ticker, years in tickers:
            for year in years:
                data[(ticker, year)] = {
                    NET_INCOME_COL: random.randint(1, 100) * 1e6,
                    TOTAL_ASSETS_COL: random.randint(1, 100) * 1e6,
                }
        return data

    def dummy_financials(self) -> pandas.DataFrame:
        data = {
            ("A.US", 2010): {"netIncome": 91000000.0, "totalAssets": 88000000.0},
            ("A.US", 2011): {"netIncome": 84000000.0, "totalAssets": 32000000.0},
            ("A.US", 2012): {"netIncome": 91000000.0, "totalAssets": 68000000.0},
            ("A.US", 2013): {"netIncome": 9000000.0, "totalAssets": 61000000.0},
            ("A.US", 2014): {"netIncome": 4000000.0, "totalAssets": 24000000.0},
            ("A.US", 2015): {"netIncome": 88000000.0, "totalAssets": 26000000.0},
            ("A.US", 2016): {"netIncome": 97000000.0, "totalAssets": 36000000.0},
            ("A.US", 2017): {"netIncome": 97000000.0, "totalAssets": 67000000.0},
            ("A.US", 2018): {"netIncome": 50000000.0, "totalAssets": 27000000.0},
            ("A.US", 2019): {"netIncome": 90000000.0, "totalAssets": 27000000.0},
            ("B.US", 2015): {"netIncome": 37000000.0, "totalAssets": 51000000.0},
            ("B.US", 2016): {"netIncome": 27000000.0, "totalAssets": 36000000.0},
            ("B.US", 2017): {"netIncome": 85000000.0, "totalAssets": 44000000.0},
            ("B.US", 2018): {"netIncome": 60000000.0, "totalAssets": 89000000.0},
            ("B.US", 2019): {"netIncome": 39000000.0, "totalAssets": 90000000.0},
            ("B.US", 2020): {"netIncome": 75000000.0, "totalAssets": 77000000.0},
            ("B.US", 2021): {"netIncome": 18000000.0, "totalAssets": 67000000.0},
            ("B.US", 2022): {"netIncome": 7000000.0, "totalAssets": 37000000.0},
            ("B.US", 2023): {"netIncome": 2000000.0, "totalAssets": 18000000.0},
            ("B.US", 2024): {"netIncome": 95000000.0, "totalAssets": 92000000.0},
        }

        financials = pandas.DataFrame.from_dict(data, orient="index")
        return financials

    def test_returns_x_y(self):
        financials = self.dummy_financials()
        dataset_x, dataset_y = RoADataLoader(financials=financials).get()

        expected_x_dict = {
            "netIncome": {
                ("A.US", 2012): 91.0,
                ("A.US", 2013): 9.0,
                ("A.US", 2014): 4.0,
                ("B.US", 2017): 85.0,
                ("B.US", 2018): 60.0,
                ("B.US", 2019): 39.0,
            },
            "totalAssets": {
                ("A.US", 2012): 68.0,
                ("A.US", 2013): 61.0,
                ("A.US", 2014): 24.0,
                ("B.US", 2017): 44.0,
                ("B.US", 2018): 89.0,
                ("B.US", 2019): 90.0,
            },
            "netIncomeRollingMean": {
                ("A.US", 2012): 88.66666666666667,
                ("A.US", 2013): 61.333333333333336,
                ("A.US", 2014): 34.666666666666664,
                ("B.US", 2017): 49.666666666666664,
                ("B.US", 2018): 57.333333333333336,
                ("B.US", 2019): 61.333333333333336,
            },
            "totalAssetsRollingMean": {
                ("A.US", 2012): 62.666666666666664,
                ("A.US", 2013): 53.666666666666664,
                ("A.US", 2014): 51.0,
                ("B.US", 2017): 43.666666666666664,
                ("B.US", 2018): 56.333333333333336,
                ("B.US", 2019): 74.33333333333333,
            },
        }
        expected_y_dict = {
            "netIncomeRollingMeanFuture": {
                ("A.US", 2012): 59.0,
                ("A.US", 2013): 67.2,
                ("A.US", 2014): 84.4,
                ("B.US", 2017): 39.8,
                ("B.US", 2018): 28.2,
                ("B.US", 2019): 39.4,
            },
            "totalAssetsRollingMeanFuture": {
                ("A.US", 2012): 42.8,
                ("A.US", 2013): 36.0,
                ("A.US", 2014): 36.6,
                ("B.US", 2017): 72.0,
                ("B.US", 2018): 57.8,
                ("B.US", 2019): 58.2,
            },
            "returnOnEquity": {
                ("A.US", 2012): 1.3785046728971964,
                ("A.US", 2013): 1.8666666666666667,
                ("A.US", 2014): 2.3060109289617485,
                ("B.US", 2017): 0.5527777777777777,
                ("B.US", 2018): 0.4878892733564014,
                ("B.US", 2019): 0.6769759450171821,
            },
        }

        self.assertEqual(expected_x_dict, dataset_x.to_dict())
        self.assertEqual(expected_y_dict, dataset_y.to_dict())
