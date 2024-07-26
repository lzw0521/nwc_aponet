import matplotlib.pyplot as plt
import re  # 导入正则表达式库

def parse_loss_from_file(file_path):
    """
    从log文件中解析损失值。
    """
    losses = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # 提取损失值并转换为浮点数
                loss = -float(line.split(':')[-1].strip())
                loss=loss/10000
                losses.append(loss)
            except ValueError:
                # 如果转换失败（例如，遇到格式不正确的行），跳过该行
                continue
    return losses


def extract_vehicle_num_from_filename(filename):
    """
    从文件名中提取学习率。
    """
    # 使用正则表达式从文件名中提取学习率
    match = re.search(r'vehicle_([0-9]+)', filename)
    if match:
        return match.group(1)  # 返回匹配到的学习率字符串
    return "Unknown vehicle num"  # 如果没有找到匹配项，则返回未知

def plot_losses(losses_list, labels):
    """
    绘制损失值变化图。
    """
    for losses, label in zip(losses_list, labels):
        plt.plot(losses, label=f'Robots_num: {label}', linewidth=0.4)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('The loss function iteration result')
    plt.legend(loc='upper right', bbox_to_anchor=(1,0.9))
    # 定义文本样式
    fontdict = {
        'fontname': 'Arial',  # 字体名
        # 'weight': 'bold',     # 字体粗细
        'size': 10,           # 字体大小
        'color': 'blue',      # 字体颜色
    }
    plt.text(0.5, 0.9, f'Number of training nodes: 100', ha='center', va='bottom', transform=plt.gca().transAxes,fontdict=fontdict)
    plt.show()


    
# log_files = [
#     '52_apo_vehicle_3_loss.txt',
#     '52_apo_vehicle_5_loss.txt',
#     '52_apo_vehicle_10_loss.txt'
# ]
log_files = [
    '100_apo_vehicle_3_loss.txt',
    '100_apo_vehicle_5_loss.txt',
    '100_apo_vehicle_10_loss.txt'
]
        
losses_list = []
labels = []

# 解析每个文件中的损失值，并提取文件名中的学习率信息作为标签
for file_path in log_files:
    losses = parse_loss_from_file(file_path)
    lr = extract_vehicle_num_from_filename(file_path)
    losses_list.append(losses)
    labels.append(lr)

# 绘制损失值变化图
plot_losses(losses_list, labels)
