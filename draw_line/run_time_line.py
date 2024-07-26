import pandas as pd
import matplotlib.pyplot as plt
# 全局字体设置
plt.rcParams.update({'font.size': 18})  # 调整为14或你希望的任何大小
# 数据准备
data = {
    "Algorithm": ["GA", "SA", "NSGA-II", "GNN+DisPN", "NWC-APONet"],
    "R=3": [198.81, 217.86, 77.49, 1.69, 1.58],
    "R=5": [277.35, 338.88, 79.68, 1.97, 2.01],
    "R=10": [315.89, 376.65, 93.39, 2.84, 2.47]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置图表大小
plt.figure(figsize=(12, 8))

# 绘制不同机器人数量的折线
markers = ['o', 's', '^']  # 不同的标记形状
for i, (key, marker) in enumerate(zip(["R=3", "R=5", "R=10"], markers)):
    plt.plot(df['Algorithm'], df[key], marker=marker, label=key)

# plt.title('Comparison of Task Completion Time Across Different Robot Counts')
# plt.xlabel('Algorithm')
plt.ylabel('Task Completion Time (s)')
plt.yscale('log')  # 使用对数尺度
plt.legend(title='Number of Robots')
plt.grid(True)

plt.show()
