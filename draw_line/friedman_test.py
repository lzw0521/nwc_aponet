import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

# 提取数据
data = {
    "GA": {
        "time_cost": [142.8103, 88.2997, 56.4495],
        "balance": [70.9409, 80.2133, 91.0851],
        "running_time": [198.81, 277.86, 315.89]
    },
    "SA": {
        "time_cost": [147.388, 89.684, 59.8975],
        "balance": [72.32, 39.5613, 55.1019],
        "running_time": [217.86, 338.88, 376.65]
    },
    "NSGA-II": {
        "time_cost": [138.2319, 83.7467, 54.8063],
        "balance": [2.784, 21.8675, 17.6547],
        "running_time": [77.49, 79.68, 93.39]
    },
    "GNN+DiSPN": {
        "time_cost": [148.3566, 96.6889, 68.6348],
        "balance": [83.62251, 132.15951, 75.494647],
        "running_time": [1.69, 1.97, 2.84]
    },
    "NWC-APONet": {
        "time_cost": [132.729, 82.16, 50.2069],
        "balance": [72.104897, 36.418043, 43.20956],
        "running_time": [1.58, 2.01, 2.47]
    }
}

# 将数据转换为DataFrame
df_time_cost = pd.DataFrame({key: value['time_cost'] for key, value in data.items()})
df_balance = pd.DataFrame({key: value['balance'] for key, value in data.items()})
df_running_time = pd.DataFrame({key: value['running_time'] for key, value in data.items()})

# 进行Friedman检验
friedman_time_cost = friedmanchisquare(*[df_time_cost[col] for col in df_time_cost])
friedman_balance = friedmanchisquare(*[df_balance[col] for col in df_balance])
friedman_running_time = friedmanchisquare(*[df_running_time[col] for col in df_running_time])

# 输出Friedman检验结果
friedman_results = pd.DataFrame({
    "Metric": ["Time Cost", "Balance", "Running Time"],
    "Friedman Statistic": [friedman_time_cost.statistic, friedman_balance.statistic, friedman_running_time.statistic],
    "p-value": [friedman_time_cost.pvalue, friedman_balance.pvalue, friedman_running_time.pvalue]
})

# 进行Wilcoxon符号秩检验，逐对比较
algorithms = list(data.keys())
wilcoxon_results = []

for i in range(len(algorithms)):
    for j in range(i+1, len(algorithms)):
        alg1 = algorithms[i]
        alg2 = algorithms[j]
        wilcoxon_time_cost = wilcoxon(df_time_cost[alg1], df_time_cost[alg2])
        wilcoxon_balance = wilcoxon(df_balance[alg1], df_balance[alg2])
        wilcoxon_running_time = wilcoxon(df_running_time[alg1], df_running_time[alg2])
        wilcoxon_results.append([f"{alg1} vs {alg2}", "Time Cost", wilcoxon_time_cost.statistic, wilcoxon_time_cost.pvalue])
        wilcoxon_results.append([f"{alg1} vs {alg2}", "Balance", wilcoxon_balance.statistic, wilcoxon_balance.pvalue])
        wilcoxon_results.append([f"{alg1} vs {alg2}", "Running Time", wilcoxon_running_time.statistic, wilcoxon_running_time.pvalue])

wilcoxon_df = pd.DataFrame(wilcoxon_results, columns=["Comparison", "Metric", "Wilcoxon Statistic", "p-value"])

# 输出结果
print("Friedman 检验结果:")
print(friedman_results)

print("\nWilcoxon 符号秩检验结果:")
print(wilcoxon_df)

# 保存结果到CSV文件
friedman_results.to_csv("friedman_test_results.csv", index=False)
wilcoxon_df.to_csv("wilcoxon_test_results.csv", index=False)

print("\n数据已保存到文件 'friedman_test_results.csv' 和 'wilcoxon_test_results.csv'")
