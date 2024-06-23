from numpy import NaN
import pandas
import operator
from functools import reduce


from typing import Any, Optional, Self, Dict, Any


class FinancialsForTicker:
    labels_balance_sheet = [
        "totalAssets",
        "intangibleAssets",
        "earningAssets",
        "otherCurrentAssets",
        "totalLiab",
        "totalStockholderEquity",
        "deferredLongTermLiab",
        "otherCurrentLiab",
        "commonStock",
        "capitalStock",
        "retainedEarnings",
        "otherLiab",
        "goodWill",
        "otherAssets",
        "cash",
        "cashAndEquivalents",
        "totalCurrentLiabilities",
        "currentDeferredRevenue",
        "netDebt",
        "shortTermDebt",
        "shortLongTermDebt",
        "shortLongTermDebtTotal",
        "otherStockholderEquity",
        "propertyPlantEquipment",
        "totalCurrentAssets",
        "longTermInvestments",
        "netTangibleAssets",
        "shortTermInvestments",
        "netReceivables",
        "longTermDebt",
        "inventory",
        "accountsPayable",
        "totalPermanentEquity",
        "noncontrollingInterestInConsolidatedEntity",
        "temporaryEquityRedeemableNoncontrollingInterests",
        "accumulatedOtherComprehensiveIncome",
        "additionalPaidInCapital",
        "commonStockTotalEquity",
        "preferredStockTotalEquity",
        "retainedEarningsTotalEquity",
        "treasuryStock",
        "accumulatedAmortization",
        "nonCurrrentAssetsOther",
        "deferredLongTermAssetCharges",
        "nonCurrentAssetsTotal",
        "capitalLeaseObligations",
        "longTermDebtTotal",
        "nonCurrentLiabilitiesOther",
        "nonCurrentLiabilitiesTotal",
        "negativeGoodwill",
        "warrants",
        "preferredStockRedeemable",
        "capitalSurpluse",
        "liabilitiesAndStockholdersEquity",
        "cashAndShortTermInvestments",
        "propertyPlantAndEquipmentGross",
        "propertyPlantAndEquipmentNet",
        "accumulatedDepreciation",
        "netWorkingCapital",
        "netInvestedCapital",
        "commonStockSharesOutstanding",
    ]
    labels_cash_flow = [
        "investments",
        "changeToLiabilities",
        "totalCashflowsFromInvestingActivities",
        "netBorrowings",
        "totalCashFromFinancingActivities",
        "changeToOperatingActivities",
        "netIncome",
        "changeInCash",
        "beginPeriodCashFlow",
        "endPeriodCashFlow",
        "totalCashFromOperatingActivities",
        "issuanceOfCapitalStock",
        "depreciation",
        "otherCashflowsFromInvestingActivities",
        "dividendsPaid",
        "changeToInventory",
        "changeToAccountReceivables",
        "salePurchaseOfStock",
        "otherCashflowsFromFinancingActivities",
        "changeToNetincome",
        "capitalExpenditures",
        "changeReceivables",
        "cashFlowsOtherOperating",
        "exchangeRateChanges",
        "cashAndCashEquivalentsChanges",
        "changeInWorkingCapital",
        "stockBasedCompensation",
        "otherNonCashItems",
        "freeCashFlow",
    ]
    labels_income_statement = [
        "researchDevelopment",
        "effectOfAccountingCharges",
        "incomeBeforeTax",
        "minorityInterest",
        "netIncome",
        "sellingGeneralAdministrative",
        "sellingAndMarketingExpenses",
        "grossProfit",
        "reconciledDepreciation",
        "ebit",
        "ebitda",
        "depreciationAndAmortization",
        "nonOperatingIncomeNetOther",
        "operatingIncome",
        "otherOperatingExpenses",
        "interestExpense",
        "taxProvision",
        "interestIncome",
        "netInterestIncome",
        "extraordinaryItems",
        "nonRecurring",
        "otherItems",
        "incomeTaxExpense",
        "totalRevenue",
        "totalOperatingExpenses",
        "costOfRevenue",
        "totalOtherIncomeExpenseNet",
        "discontinuedOperations",
        "netIncomeFromContinuingOps",
        "netIncomeApplicableToCommonShares",
        "preferredStockAndOtherAdjustments",
    ]
    all_statement_labels = (
        labels_balance_sheet + labels_cash_flow + labels_income_statement
    )
    all_rolling_labels = [f"{label}3YrMean" for label in all_statement_labels]
    all_labels = ["id", "year"] + all_statement_labels + all_rolling_labels

    @classmethod
    def from_db(cls, document) -> Optional[Self]:
        if "Financials" not in document:
            return None

        balance_sheet = document["Financials"]["Balance_Sheet"]["yearly"]
        cash_flow = document["Financials"]["Cash_Flow"]["yearly"]
        income_statement = document["Financials"]["Income_Statement"]["yearly"]
        reports = [balance_sheet, cash_flow, income_statement]

        def parse_year(date: str) -> int:
            return int(date.split("-")[0])

        id: str = document["_id"]
        dates = [date for report in reports for date in report.keys()]
        years = sorted(set(map(parse_year, dates)))
        data: Dict[tuple[Any, int], dict[str, str | float | None]] = {
            (id, year): {
                label: NaN for label in FinancialsForTicker.all_statement_labels
            }
            for year in years
        }

        for report, statement_labels in [
            (balance_sheet, FinancialsForTicker.labels_balance_sheet),
            (cash_flow, FinancialsForTicker.labels_cash_flow),
            (income_statement, FinancialsForTicker.labels_income_statement),
        ]:
            for date, statement in report.items():
                # only include statements reported in USD
                if statement["currency_symbol"] != "USD":
                    continue
                year = parse_year(date)

                for label in statement_labels:
                    value = statement[label]
                    # only use non-null value for duplicate reports of the same year
                    if label in data[(id, year)] and value is None:
                        continue
                    data[(id, year)][label] = float(value) if value else None

        dataframe = pandas.DataFrame.from_dict(data, orient="index").astype("float")
        dataframe_3yr_mean = pandas.DataFrame(dataframe.rolling(3).mean())
        dataframe_3yr_mean = dataframe_3yr_mean.rename(
            columns=lambda col: f"{col}3YrMean"
        )
        dataframe = dataframe.merge(
            dataframe_3yr_mean, left_index=True, right_index=True
        )

        return cls(dataframe)

    @classmethod
    def to_csv_header(cls) -> str:
        return f"{','.join(FinancialsForTicker.all_labels)}\n"

    def __init__(self, dataframe: pandas.DataFrame) -> None:
        self.dataframe = dataframe

    def __getitem__(self, key: str) -> Any:
        return self.dataframe[key]

    def to_csv(self) -> str:
        def get_value(id: str, year: str, row, label: str) -> str:
            if label == "year":
                return year
            if label == "id":
                return id
            return str(getattr(row, label))

        def to_csv_line(row) -> str:
            id = getattr(row, "Index")[0]
            year = str(getattr(row, "Index")[1])
            values = map(
                lambda label: get_value(id, year, row, label),
                FinancialsForTicker.all_labels,
            )
            return f"{','.join(values)}\n"

        lines = map(to_csv_line, self.dataframe.itertuples())

        return reduce(operator.add, lines, "")
