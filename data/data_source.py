from pathlib import Path
from pymongo import MongoClient
from pymongo.cursor import Cursor
from tqdm import tqdm

from typing import Any

from data.ticker import Ticker, TickerList
from data.financials import FinancialsForTicker


class DataDB:
    def __init__(self):
        uri = "mongodb://localhost:27017"
        client = MongoClient(uri)
        self.DB = "stocks"
        self.FUNDAMENTALS = "fundamentals"
        self.db = client[self.DB]

    def get_US_stocks_with_financial_reports(self) -> Cursor:
        query = {
            "Financials.Balance_Sheet.yearly": {"$ne": None},
            "General.CountryISO": "US",
            "General.Type": "Common Stock",
        }
        return self.db[self.FUNDAMENTALS].find(query)

    def get_financials_by_id(self, id) -> Any:
        query = {
            "_id": id,
            "Financials.Balance_Sheet.yearly": {"$ne": None},
        }
        return self.db[self.FUNDAMENTALS].find_one(query)


class DataCsv:
    def __init__(self):
        self.data_db = DataDB()
        self.tickers_csv = self.get_file("tickers.csv")
        self.financials_csv = self.get_file("financials.csv")

    def get_file(self, filename: str) -> Path:
        return Path(__file__).with_name(filename)

    def save_tickers(self) -> None:
        with self.tickers_csv.open("w") as file:
            file.write(Ticker.to_csv_header())

            fundamentals = self.data_db.get_US_stocks_with_financial_reports()

            for record in tqdm(fundamentals):
                ticker = Ticker.from_db(record)
                if ticker and ticker["Exchange"] in [
                    "NYSE",
                    "NASDAQ",
                    "NYSE ARCA",
                ]:
                    file.write(ticker.to_csv())

    def save_financials(self) -> None:
        with self.financials_csv.open("w") as file:
            file.write(FinancialsForTicker.to_csv_header())

            tickers = TickerList.from_csv(self.tickers_csv.as_posix())

            for id in tqdm(tickers["id"]):
                fundamentals = self.data_db.get_financials_by_id(id)
                financials = FinancialsForTicker.from_db(fundamentals)
                if financials:
                    file.write(financials.to_csv())
