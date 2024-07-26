from gin import Net
import torch.nn as nn
import torch.nn.functional as F
import torch
import math 
from torch.distributions import Categorical #从PyTorch中导入Categorical分布，用于采样动作
from ortools_tsp import solve #从名为ortools_tsp的模块中导入solve函数
import subprocess
import os
import datetime
import matplotlib.pyplot as plt

def normalize_data(data):
    # 分别提取x坐标、y坐标和工作量
    x_coords = [item[0] for item in data]
    y_coords = [item[1] for item in data]
    workloads = [item[2] for item in data]
    
    # 分别计算每个维度的最小值和最大值
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    workloads_min, workloads_max = min(workloads), max(workloads)
    
    # 归一化数据
    normalized_data = []
    for x, y, workload in data:
        x_normalized = (x - x_min) / (x_max - x_min)
        y_normalized = (y - y_min) / (y_max - y_min)
        workload_normalized = (workload - workloads_min) / (workloads_max - workloads_min)
        normalized_data.append([x_normalized, y_normalized, workload_normalized])
    
    return normalized_data


#log_gnn_dispn/log_lam5
def append_loss_to_file(loss, folder_path='./log/node100_log_10_apo_t_w_lam'):
    # 获取当前日期并格式化为字符串，例如 '2023-09-17'
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"loss_{current_date}.txt"
    full_path = f"{folder_path}/{file_name}"

    # 追加损失值到文件（如果文件不存在，则创建）
    with open(full_path, 'a') as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Loss at {current_time}: {loss}\n")

def append_valited_result_to_file(valited_result, folder_path='./log/node100_log_10_apo_t_w_lam'):
    # 获取当前日期并格式化为字符串，例如 '2023-09-17'
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"valited_result_{current_date}.txt"
    full_path = f"{folder_path}/{file_name}"

    # 追加损失值到文件（如果文件不存在，则创建）
    with open(full_path, 'a') as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"valited_result {current_time}: {valited_result}\n")
        
def append_reward0_to_file(reward0, folder_path='./log/node100_log_10_apo_t_w_lam'):
    # 获取当前日期并格式化为字符串，例如 '2023-09-17'
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"reward0_{current_date}.txt"
    full_path = f"{folder_path}/{file_name}"

    # 追加损失值到文件（如果文件不存在，则创建）
    with open(full_path, 'a') as file:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"reward0 {current_time}: {reward0}\n")

def balance_workload(actions, data, num_agents, max_nodes_per_agent=None):
    data = data[:, 1:, :]
    batch_size, num_nodes, _ = data.shape
    optimized_actions = actions.clone()
    print("---->enter balance_workload")
    if max_nodes_per_agent is None:
        max_nodes_per_agent = -(-num_nodes // num_agents)  # 向上取整

    for batch in range(batch_size):
        # 初始化代理工作量统计
        agent_workloads = torch.zeros(num_agents)
        
        # 计算每个代理的初始工作量
        for agent in range(num_agents):
            agent_nodes = (optimized_actions[batch] == agent).nonzero(as_tuple=True)[0]
            # 避免超出索引，确保node_idx在data的范围内
            for node_idx in agent_nodes:
                if node_idx < num_nodes:
                    _, _, workload = data[batch, node_idx].tolist()
                    agent_workloads[agent] += workload / 1000
        
        # 尝试平衡工作量
        for i in range(num_nodes):
            current_agent = optimized_actions[batch, i]
            # 寻找工作量最小的代理
            min_workload_agent = torch.argmin(agent_workloads)
            # 确保当前节点可以被重新分配
            if agent_workloads[current_agent] > agent_workloads[min_workload_agent] + data[batch, i, 2] / 1000:
                # 更新工作量
                agent_workloads[current_agent] -= data[batch, i, 2] / 1000
                agent_workloads[min_workload_agent] += data[batch, i, 2] / 1000
                # 重新分配节点
                optimized_actions[batch, i] = min_workload_agent

    # 注意：sub_operate_distance没有在此函数中更新，因为它需要更多上下文来确定如何计算
    # 如果您需要计算每个代理的操作距离，请确保提供适当的逻辑来更新sub_operate_distance

    return optimized_actions

def postprocess_actions(actions, distance_matrices,num_agents, threshold,max_nodes_per_agent=None):
    batch_size, num_nodes = actions.shape
    optimized_actions = actions.clone()
    max_nodes_per_agent = -(-num_nodes // num_agents)  # 向上取整
    # max_nodes_per_agent = num_nodes // num_agents#向下取整
    distance_matrix=torch.zeros(num_nodes, num_nodes)
    for batch in range(batch_size):
        distance_matrix = distance_matrices[batch]
        agent_node_count = torch.bincount(optimized_actions[batch], minlength=optimized_actions.max() + 1)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if optimized_actions[batch, i] != optimized_actions[batch, j] and distance_matrix[i, j] < threshold:
                    # 检查是否可以重新分配
                    if max_nodes_per_agent is None or agent_node_count[optimized_actions[batch, i]] < max_nodes_per_agent:
                        agent_node_count[optimized_actions[batch, j]] -= 1
                        agent_node_count[optimized_actions[batch, i]] += 1
                        optimized_actions[batch, j] = optimized_actions[batch, i]

    return optimized_actions



def plot_sub_tours(sub_tours):  
    """
    在2D平面上绘制每个子巡回路径。
    
    :param sub_tours: 一个列表，包含多个子巡回路径，每个子巡回路径是由 (x, y) 坐标组成的列表。
    """
    plt.figure(figsize=(10, 10))  # 创建一个新的图形，指定大小

    # 创建一个颜色列表
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_index = 0

    for i, sub_tour in enumerate(sub_tours):
        # 解压 (x, y) 坐标
        x, y = zip(*sub_tour)

        # 将起点和终点相连，形成闭合路径
        x += (x[0],)
        y += (y[0],)

         # 绘制子巡回路径，每个子巡回路径使用不同的颜色
        plt.plot(x, y, marker='o', color=colors[color_index], label=f'Robot {i+1}')
         # 更新颜色索引
        color_index = (color_index + 1) % len(colors)

    # 设置图表标题和坐标轴标签
    plt.title('Agricultural multi-robots Task allocation ')
    plt.xlabel('Coordinate-X')
    plt.ylabel('Coordinate-Y')

    # 添加图例，以区分不同的子巡回路径
    plt.legend()

    # 显示图表
    plt.show()

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(line1, line2):
    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def check_intersections_between_tours(adjusted_sub_tours):
    intersection_count = 0
    
    # 遍历所有的旅行商路线对
    for i in range(len(adjusted_sub_tours)):
        for j in range(i + 1, len(adjusted_sub_tours)):
            # 对于每对旅行商路线，检查它们的每对线段是否相交
            for a in range(len(adjusted_sub_tours[i]) - 1):
                line1 = (adjusted_sub_tours[i][a], adjusted_sub_tours[i][a + 1])
                for b in range(len(adjusted_sub_tours[j]) - 1):
                    line2 = (adjusted_sub_tours[j][b], adjusted_sub_tours[j][b + 1])
                    if intersect(line1, line2):
                        intersection_count += 1  # 发现交点，计数器加1

    return intersection_count


# 定义欧几里得距离计算函数
def euclidean_distance(coord1, coord2):
    # print(f"coord1:{coord1}")
    # print(f"coord2:{coord2}")
    return torch.sqrt(torch.sum((coord1 - coord2) ** 2, dim=-1))

def calculate_threshold(distance_matrices, percentile=50):
    """
    根据距离矩阵的统计特性计算阈值。

    :param distance_matrices: 距离矩阵的张量。
    :param percentile: 用于计算阈值的百分位数。
    :return: 计算得到的阈值。
    """
    distances = distance_matrices[distance_matrices > 0]  # 排除零距离（即节点自身）
    threshold = torch.quantile(distances, percentile / 100.0)
    return threshold.item()
# def calculate_threshold(distance_matrices, percentile=50, sample_size=10000):
#     # 扁平化距离矩阵并去除零距离
#     distances = distance_matrices.view(-1)
#     distances = distances[distances > 0]

#     # 随机抽样
#     if distances.numel() > sample_size:
#         indices = torch.randint(0, distances.size(0), (sample_size,))
#         sampled_distances = distances[indices]
#     else:
#         sampled_distances = distances

#     # 计算抽样数据的分位数
#     threshold = torch.quantile(sampled_distances, percentile / 100.0).item()
#     return threshold
def compute_distance_matrices(fea):
    # 计算每个批次的距离矩阵
    n_batch, n_nodes, _ = fea.shape
    distance_matrices = torch.zeros((n_batch, n_nodes, n_nodes))

    for batch in range(n_batch):
        # 扩展维度以便进行广播
        coords1 = fea[batch].unsqueeze(1)  # 形状变为 [n_nodes, 1, 3]
        coords2 = fea[batch].unsqueeze(0)  # 形状变为 [1, n_nodes, 3]

        # 计算两两节点间的距离
        distances = torch.sqrt(torch.sum((coords1 - coords2) ** 2, dim=2))
        distance_matrices[batch] = distances

    return distance_matrices

def parse_tsp(filename):
    """解析.tsp文件，提取坐标"""
    coordinates = []
    with open(filename, 'r') as file:
        start_reading = False
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                start_reading = True
                continue
            elif line.strip() == "EOF":
                break
            if start_reading:
                _, x, y = line.strip().split()
                coordinates.append([float(x), float(y)])
    return coordinates

def parse_tour(filename):
    """解析.opt.tour文件，提取工作量"""
    workload = []
    with open(filename, 'r') as file:
        start_reading = False
        for line in file:
            if line.strip() == "TOUR_SECTION":
                start_reading = True
                continue
            elif line.strip() == "-1" or line.strip() == "EOF":
                break
            if start_reading:
                workload.append(float(line.strip()))
    # 生成工作量（这里简化为顺序编号，您可以根据需要调整）
    #workload = [i for i in range(1, len(tour) + 1)]
    return workload

#最小最大归一化
def min_max_normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in values]

# Z-Score 归一化
def z_score_normalize(values):
    mean_val = sum(values) / len(values)
    std_val = (sum([(val - mean_val) ** 2 for val in values]) / len(values)) ** 0.5
    return [(val - mean_val) / std_val if std_val != 0 else 0 for val in values]

#lkh3
def solve_tsp_with_lkh(nodes):
    """
    解决TSP问题使用LKH-3算法。
    
    :param nodes: 旅行商问题的节点列表，格式为 (x, y) 坐标。
    :param lkh_path: LKH-3可执行文件的路径。
    :param temp_dir: 用于存储临时文件的目录。
    :return: 最短路径的长度。
    """
    lkh_path = "./LKH-3.0.9/LKH"  # 将这里替换为你的LKH-3可执行文件路径
    temp_dir = "./LKH-3.0.9/directory"  # 临时文件目录
    # 创建LKH输入文件
    problem_file = os.path.join(temp_dir, "tsp_problem.tsp")
    with open(problem_file, "w") as file:
        file.write("NAME: TSP\n")
        file.write("TYPE: TSP\n")
        file.write("DIMENSION: {}\n".format(len(nodes)))
        file.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        file.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(nodes, start=1):
            file.write("{} {} {}\n".format(i, x, y))
        file.write("EOF\n")

    # 创建LKH参数文件
    param_file = os.path.join(temp_dir, "lkh_param.par")
    with open(param_file, "w") as file:
        file.write("PROBLEM_FILE = {}\n".format(problem_file))
        file.write("OUTPUT_TOUR_FILE = {}\n".format(os.path.join(temp_dir, "tsp_solution.tour")))

    # 调用LKH-3
    # subprocess.run([lkh_path, param_file])
    subprocess.run([lkh_path, param_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 解析解决方案文件以获取路径长度
    solution_file = os.path.join(temp_dir, "tsp_solution.tour")
    with open(solution_file, "r") as file:
        lines = file.readlines()
     # 在LKH输出文件中找到路径长度
        length_line = [line for line in lines if "Length =" in line]
        if length_line:
            # 提取路径长度
            length_str = length_line[0].split('=')[1].strip()
            length = int(length_str)
        else:
            
            length = None

    return length

# 示例用法
# nodes = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160), (100, 160), (200, 160), (140, 140), (40, 120), (100, 120)]
# print(f"nodes:{nodes}")
# print(solve_tsp_with_lkh(nodes))
