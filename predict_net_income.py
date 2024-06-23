from matplotlib import pyplot as plt

from models.select_random import SelectRandom
from scenarios.fixed_period_net_income import FixedPeriodNetIncomeScenario
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.select_top import SelectTop


scenario = FixedPeriodNetIncomeScenario(2010, 2015)

# # Decision Tree almost always performs worse than random forest
# decision_tree_model = DecisionTree(max_leaf_nodes=18)
# decision_tree_metric = scenario.train(decision_tree_model)
# print(decision_tree_metric)
# decision_tree_model.visualize()
#
random_forest_model = RandomForest()
random_forest_metric = scenario.train(random_forest_model)
print(random_forest_metric)

# # Benchmark select top 50% performing tickers
select_top_model = SelectTop(
    frac=0.5, sorted_predictions=scenario.sorted_predictions(), ascending=False
)
select_top_metric = scenario.train(select_top_model)
print(select_top_metric)

select_random_model = SelectRandom(frac=0.5, predictions=scenario.sorted_predictions())
select_random_metric = scenario.train(select_random_model)
print(select_random_metric)

plt.plot(
    ["random forest", "select top", "select random"],
    [
        m.value()
        for m in [random_forest_metric, select_top_metric, select_random_metric]
    ],
)
plt.show()

# epochs = 10
#
# max_leaf_nodes_list = range(17, 40)
# scenarios = [scenario]
# models = [
#     DecisionTree(max_leaf_nodes=max_leaf_nodes)
#     for max_leaf_nodes in max_leaf_nodes_list
# ]
#
# result = [
#     (scenario.train_mean_metric(model), scenario, model)
#     for model in models
#     for scenario in scenarios
# ]
#
# sorted_result = sorted(result, key=lambda x: x[0], reverse=True)
#
# for metric_mean, scenario, model in sorted_result:
#     print(scenario)
#     print(model)
#     print(f"Mean Metric: {metric_mean:.2f}%")
#
#
# plt.plot(max_leaf_nodes_list, [m for m, _, _ in result])
# plt.show()
