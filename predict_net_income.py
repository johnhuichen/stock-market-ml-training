from typing import Tuple

from pandas import pandas

from data_loader.future_net_income import FutureNetIncomeDataLoader
from metrics.future_net_income import NetIncomeMetric
from models.decision_tree_classifier import DecisionTreeClassiferModel
from models.random_forest_classifier import RandomForestClassifierModel
from models.select_random import SelectRandomModel
from models.select_top import SelectTopModel
from trainer.trainer import Trainer


def get_metric(trainer, model, future_net_incomes):
    predictions, val_y = trainer.train(model)
    return NetIncomeMetric(
        model=model,
        predictions=predictions,
        val_y=val_y,
        future_net_incomes=future_net_incomes,
    )


def predict_fixed_period_exceeds_threshold(
    year: int, threshold: float
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    dataloader = FutureNetIncomeDataLoader()
    dataset_x, future_net_incomes = dataloader.get()

    dataset_x = dataset_x.xs(year, level=1)
    future_net_incomes = future_net_incomes.xs(year, level=1)

    dataset_y = (
        future_net_incomes.loc[:, [FutureNetIncomeDataLoader.RETURN_FUTURE]] > threshold
    )
    return dataset_x, dataset_y, future_net_incomes


def predict_any_period_exceeds_threshold(
    threshold: float,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    dataloader = FutureNetIncomeDataLoader()
    dataset_x, future_net_incomes = dataloader.get()

    dataset_y = (
        future_net_incomes.loc[:, [FutureNetIncomeDataLoader.RETURN_FUTURE]] > threshold
    )
    return dataset_x, dataset_y, future_net_incomes


# dataset_x, dataset_y, future_net_incomes = predict_fixed_period_exceeds_threshold(
#     year=2016, threshold=0.15
# )
dataset_x, dataset_y, future_net_incomes = predict_any_period_exceeds_threshold(0.15)

trainer = Trainer(dataset_x, dataset_y)

decision_tree_model = DecisionTreeClassiferModel(max_leaf_nodes=5)
random_forest_model = RandomForestClassifierModel()
select_top_model = SelectTopModel(
    frac=0.5,
    cheatsheet=future_net_incomes,
    sort_by_col=FutureNetIncomeDataLoader.RETURN_FUTURE,
    ascending=False,
)
select_random_model = SelectRandomModel(0.5)

print(get_metric(trainer, select_top_model, future_net_incomes))
print(get_metric(trainer, select_random_model, future_net_incomes))
print(get_metric(trainer, decision_tree_model, future_net_incomes))
print(get_metric(trainer, random_forest_model, future_net_incomes))
# decision_tree_model.visualize()
