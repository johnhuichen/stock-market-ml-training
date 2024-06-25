from data_loader.future_net_income import FutureNetIncomeDataLoader
from metrics.future_net_income import NetIncomeMetric
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.select_random import SelectRandom
from models.select_top import SelectTop
from trainer.trainer import Trainer

dataloader = FutureNetIncomeDataLoader()
dataset_x, future_net_incomes = dataloader.get()
# dataset_x = dataset_x.xs(2016, level=1)
# future_net_incomes = future_net_incomes.xs(2016, level=1)

# Predict if net income exceeds 10%
threshold = 0.1
dataset_y = (
    future_net_incomes.loc[:, [FutureNetIncomeDataLoader.RETURN_FUTURE]] > threshold
)

trainer = Trainer(dataset_x, dataset_y)


def get_metric(model):
    predictions, val_y = trainer.train(model)
    return NetIncomeMetric(
        model=model,
        predictions=predictions,
        val_y=val_y,
        future_net_incomes=future_net_incomes,
    )


decision_tree_model = DecisionTree(max_leaf_nodes=5)
random_forest_model = RandomForest()
select_top_model = SelectTop(
    frac=0.5,
    cheatsheet=future_net_incomes,
    sort_by_col=FutureNetIncomeDataLoader.RETURN_FUTURE,
    ascending=False,
)
select_random_model = SelectRandom(0.5)

print(get_metric(select_top_model))
print(get_metric(select_random_model))
print(get_metric(random_forest_model))
print(get_metric(decision_tree_model))
decision_tree_model.visualize()
