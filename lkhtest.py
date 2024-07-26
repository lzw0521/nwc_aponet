import subprocess
import os

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
    subprocess.run([lkh_path, param_file])

    # 解析解决方案文件以获取路径长度
    solution_file = os.path.join(temp_dir, "tsp_solution.tour")
    with open(solution_file, "r") as file:
        lines = file.readlines()
     # 在LKH输出文件中找到路径长度
        length_line = [line for line in lines if "Cost.min" in line]
        if length_line:
            # 提取路径长度
            length_str = length_line[0].split('=')[1].strip()
            length = int(length_str)
        else:
            length = None

    return length
# 示例用法
nodes = [(60, 200), (180, 200), (80, 180), (140, 180), (20, 160), (100, 160), (200, 160), (140, 140), (40, 120), (100, 120)]

print(solve_tsp_with_lkh(nodes))

