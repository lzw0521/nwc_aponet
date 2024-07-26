
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # 确保导入 NumPy

# 创建一个DataFrame来模拟你提供的数据
data = {
    "GA": [142.8103, 88.2907, 56.4495],
    "SA": [147.388, 89.684, 59.875],
    "NSGA-II": [138.2319, 83.7467, 54.8603],
    "GNN+DisPN": [148.3566, 96.869, 68.6348],
    "NWC-APONet": [132.729, 82.16, 50.8269]
}
df = pd.DataFrame(data, index=["R=3", "R=5", "R=10"])

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 为了在x轴上并排显示柱状图，我们需要定义每个柱的位置
n = len(df.columns)  # 算法数量
width = 0.8 / n  # 柱宽
x = np.arange(len(df))  # x轴位置数组

# 绘制柱状图
for i, column in enumerate(df.columns):
    axes[0].bar(x - width*(n/2 - i), df[column], width, label=column)

axes[0].set_title('Comparison of task completion time results (Bar Chart)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df.index)
axes[0].set_xlabel('Number of robots')
axes[0].set_ylabel('Task completion time (s)')
axes[0].legend()

# 绘制折线图，包含所有算法
for column in df.columns:
    axes[1].plot(df.index, df[column], marker='o', label=column)

axes[1].set_title('Comparison of task completion time results (Line Chart)')
axes[1].set_xlabel('Number of robots')
axes[1].set_ylabel('Task completion time (s)')
axes[1].legend()

plt.tight_layout()
plt.show()
