import torch
from common import parse_tour,parse_tsp

def data_gen(no_nodes, batch_size, flag, tsp_data):
    if flag == 'tsp' and tsp_data is not None:
         # 生成随机数据
        data = torch.rand(size=[batch_size, no_nodes, 3])
        # data = torch.randint(low=10, high=10000, size=[batch_size, n_nodes, 3]).float()

        # if tsp_data is not None and len(tsp_data) == no_nodes:
        #     # 确保 tsp_data 是一个三维数据的列表
        #     # 替换第一批数据
        #     data[0] = torch.tensor(tsp_data)
        # # 打印第一批数据和长度
        # print("第一批数据:", data[0])
        # print("数据长度:", len(data[0]))
        # # 打印批次数量
        # print("批次数量:", data.size(0))
        # 保存数据
        torch.save(data, './validation_data/validation_data6_'+str(no_nodes)+'_'+str(batch_size)+'0')
        
    elif flag == 'validation':
        # 现有的随机数据生成逻辑
        torch.save(torch.rand(size=[batch_size, no_nodes, 3]), './validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size))
    elif flag == 'testing':
        # 现有的随机数据生成逻辑
        torch.save(torch.rand(size=[batch_size, no_nodes, 3]), './testing_data/testing_data_'+str(no_nodes)+'_'+str(batch_size))
    else:
        print('flag should be "testing", "validation", or "tsp".')

if __name__ == '__main__':
    # TSP数据的加载和解析逻辑
    coordinates = parse_tsp('edata/berlin52.tsp')
    workloads = parse_tour('edata/berlin52.opt.tour')
    fea_test = [coord + [workloads[i]] for i, coord in enumerate(coordinates)]
    print(f"fea_test:{fea_test}")
    # 使用TSP数据生成
    n_nodes = 100
    b_size = 512
    flag = 'tsp'
    torch.manual_seed(3)

    data_gen(n_nodes, b_size, flag, fea_test)










# def data_gen(no_nodes, batch_size, flag):
#     if flag == 'validation':
#         torch.save(torch.rand(size=[batch_size, no_nodes, 3]), './validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size))
#     elif flag == 'testing':
#         torch.save(torch.rand(size=[batch_size, no_nodes, 3]), './testing_data/testing_data_'+str(no_nodes)+'_'+str(batch_size))
#     else:
#         print('flag should be "testing", or "validation".')

# if __name__ == '__main__':
#     n_nodes = 52
#     b_size = 512
#     flag = 'validation'
#     torch.manual_seed(3)

    

#     data_gen(n_nodes, b_size, flag)