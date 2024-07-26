from gin import Net
import torch.nn as nn
import torch.nn.functional as F
import torch
import math 
from torch.distributions import Categorical #从PyTorch中导入Categorical分布，用于采样动作
from ortools_tsp import solve #从名为ortools_tsp的模块中导入solve函数
from common import parse_tsp,parse_tour,plot_sub_tours,normalize_data,balance_workload
import statistics


class Agentembedding(nn.Module):
    def __init__(self, node_feature_size, key_size, value_size):
        super(Agentembedding, self).__init__()
        self.key_size = key_size
        self.q_agent = nn.Linear(2 * node_feature_size, key_size)
        self.k_agent = nn.Linear(node_feature_size, key_size)
        self.v_agent = nn.Linear(node_feature_size, value_size)

    def forward(self, f_c, f):
        q = self.q_agent(f_c)
        k = self.k_agent(f)
        v = self.v_agent(f)
        u = torch.matmul(k, q.transpose(-1, -2)) / math.sqrt(self.key_size)
        u_ = F.softmax(u, dim=-2).transpose(-1, -2)
        agent_embedding = torch.matmul(u_, v)

        return agent_embedding


class AgentAndNode_embedding(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size, value_size, dev):
        super(AgentAndNode_embedding, self).__init__()

        self.n_agent = n_agent

        # gin
        self.gin = Net(in_chnl=in_chnl, hid_chnl=hid_chnl).to(dev)
        # agent attention embed
        self.agents = torch.nn.ModuleList()
        for i in range(n_agent):
            self.agents.append(Agentembedding(node_feature_size=hid_chnl, key_size=key_size, value_size=value_size).to(dev))

    def forward(self, batch_graphs, n_nodes, n_batch):

        # get node embedding using gin
        nodes_h, g_h = self.gin(x=batch_graphs.x, edge_index=batch_graphs.edge_index, batch=batch_graphs.batch)
        nodes_h = nodes_h.reshape(n_batch, n_nodes, -1)
        g_h = g_h.reshape(n_batch, 1, -1)

        depot_cat_g = torch.cat((g_h, nodes_h[:, 0, :].unsqueeze(1)), dim=-1)
        # output nodes embedding should not include depot, refer to paper: https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445
        nodes_h_no_depot = nodes_h[:, 1:, :]

        # get agent embedding
        agents_embedding = []
        for i in range(self.n_agent):
            agents_embedding.append(self.agents[i](depot_cat_g, nodes_h_no_depot))

        agent_embeddings = torch.cat(agents_embedding, dim=1)

        return agent_embeddings, nodes_h_no_depot


class Policy(nn.Module):
    def __init__(self, in_chnl, hid_chnl, n_agent, key_size_embd, key_size_policy, val_size, clipping, dev):
        super(Policy, self).__init__()
        self.c = clipping
        self.key_size_policy = key_size_policy
        self.key_policy = nn.Linear(hid_chnl, self.key_size_policy).to(dev)
        self.q_policy = nn.Linear(val_size, self.key_size_policy).to(dev)

        # embed network
        self.embed = AgentAndNode_embedding(in_chnl=in_chnl, hid_chnl=hid_chnl, n_agent=n_agent,
                                            key_size=key_size_embd, value_size=val_size, dev=dev)

    def forward(self, batch_graph, n_nodes, n_batch):

        agent_embeddings, nodes_h_no_depot = self.embed(batch_graph, n_nodes, n_batch)

        k_policy = self.key_policy(nodes_h_no_depot)
        q_policy = self.q_policy(agent_embeddings)
        u_policy = torch.matmul(q_policy, k_policy.transpose(-1, -2)) / math.sqrt(self.key_size_policy)
        imp = self.c * torch.tanh(u_policy)
        prob = F.softmax(imp, dim=-2)

        return prob


def action_sample(pi):
    dist = Categorical(pi.transpose(2, 1))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action, log_prob


# def get_reward(action, data, n_agent):
  
#     subtour_max_lengths = [0 for _ in range(data.shape[0])]
#     data = data * 1000
#     depot = data[:, 0, :].tolist()
#     sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]
#     for i in range(data.shape[0]):
#         for n, m in zip(action.tolist()[i], data.tolist()[i][1:]):
#             sub_tours[i][n].append(m)
#         for tour in sub_tours[i]:
#             tour.append(depot[i])

#     for k in range(data.shape[0]):
#         for a in range(n_agent):
#             instance = sub_tours[k][a]
#             sub_tour_length,_ = solve(instance)
#             sub_tour_length= sub_tour_length/1000
#             if sub_tour_length >= subtour_max_lengths[k]:
#                 subtour_max_lengths[k] = sub_tour_length
#     return subtour_max_lengths

def get_reward(action, data, n_agent):
    print("---->enter get_reward")
    total_trans_distances = [0 for _ in range(data.shape[0])]
    all_num_intersections = [0 for _ in range(data.shape[0])]
    sub_max_trans_distance = [0 for _ in range(data.shape[0])]
    last_reward= [0 for _ in range(data.shape[0])]

    average_operate_distances = [0 for _ in range(data.shape[0])]
    sum_operate_distances = [0 for _ in range(data.shape[0])]#
    difference_operate_distances=[[0 for _ in range(n_agent)] for _ in range(data.shape[0])]#工作量与平均工作量的差值
    sum_difference_operate_distances=[0 for _ in range(data.shape[0])]#差值的累加和

    
    sub_operate_distance = [[0 for _ in range(n_agent)] for _ in range(data.shape[0])]
    sub_operate_time = [[0 for _ in range(n_agent)] for _ in range(data.shape[0])]
    subtour_trans_lengths = [[0 for _ in range(n_agent)] for _ in range(data.shape[0])]
    subtour_trans_time= [[0 for _ in range(n_agent)] for _ in range(data.shape[0])]

    total_trans_operate_distance = [[0 for _ in range(n_agent)] for _ in range(data.shape[0])]
    total_trans_operate_time = [[0 for _ in range(n_agent)] for _ in range(data.shape[0])]

    sub_max_trans_operate_distance= [0 for _ in range(data.shape[0])]
    sub_max_trans_operate_time= [0 for _ in range(data.shape[0])]
    data = data * 1000
    depot = data[:, 0, :2].tolist()
    # print(f"  depot : {depot}")
   
    #根据策略结果，确定分配方案
    sub_tours = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]#存储每个机器人分配的对应的每个地块的坐标点位
    sub_every_operate = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]#存储每个机器人对应的每个地块的工作量
    sub_ortool_tour = [[[] for _ in range(n_agent)] for _ in range(data.shape[0])]#存储每个机器人
    
    for i in range(data.shape[0]):
        for n, (x, y, workload) in zip(action.tolist()[i], data.tolist()[i][1:]):
            m = [x, y]  # 节点坐标
            sub_tours[i][n].append(m)
            # sub_every_operate[i][n].append(workload/1000)
            sub_operate_distance[i][n] += workload/1000
        for tour in sub_tours[i]:
            tour.insert(0, depot[i])
  

    #打印每个旅行商的工作量，以及工作量标准差
    for k in range(data.shape[0]):
        # print(f"Batch {k}:")
        print(f"Batch {k}, Agent  work load: {sub_operate_distance[k]}")
        std_dev = statistics.stdev(sub_operate_distance[k])
        print(f"Batch {k}, Agent  work load std_dev :{std_dev}")

    #   # 打印每个旅行商的分配方案
    # print(f"  sub_operate_distance : {sub_operate_distance}")
    # for k in range(data.shape[0]):
    #      print(f"Batch {k}:")
    #      for a in range(n_agent):
    #          instance = sub_tours[k][a]
    #          print(f"  Agent {a}: {instance}")
    
    
    #计算子旅行商转移距离
    for k in range(data.shape[0]):
        # print(f"Batch {k}:")
        for a in range(n_agent):
            instance = sub_tours[k][a]
            if len(instance) == 0:
                subtour_length=0
                sub_ortool_tour[k][a]=0
            else:
                #计算单个旅行商的长度
                subtour_length,sub_ortool_tour[k][a] = solve(instance)
                subtour_trans_lengths[k][a] = subtour_length/1000
                #除以速度换算成时间
                subtour_trans_time[k][a] = subtour_trans_lengths[k][a]/20

                total_trans_distances[k] += subtour_length/1000
    
     #计算每个批次每个机器人的(总的转移距离与总的作业距离)和       
    for k in range(data.shape[0]):
        # print(f"Batch {k}:")
        for a in range(n_agent):
            sub_operate_time[k][a]=sub_operate_distance[k][a]/5

     #计算工作量均衡程度
    for k in range(data.shape[0]):
        # print(f"Batch {k}:")
        for a in range(n_agent):
            sum_operate_distances[k]+=sub_operate_distance[k][a]
        average_operate_distances[k]=sum_operate_distances[k]/n_agent
        for a in range(n_agent):
            difference_operate_distances[k][a]=abs(sub_operate_distance[k][a]-average_operate_distances[k])
        for a in range(n_agent):
            sum_difference_operate_distances[k]+=difference_operate_distances[k][a]

    # #绘制分配结果
    # # print(sub_ortool_tour[0])
    # # print(f"  sub_ortool_tour[0]: {sub_ortool_tour[0]}")
    # # 调整数据结构
    # adjusted_sub_tours = [tour[0] for tour in sub_ortool_tour[0]]  # 从四层嵌套降到二层
    # # print(f"  adjusted_sub_tours: {adjusted_sub_tours}")
    # plot_sub_tours(adjusted_sub_tours)

    # #计算每个批次中 每个旅行商运动路线的交点个数
    # for k in range(data.shape[0]):
    #     adjusted_sub_tours = [tour[0] for tour in sub_ortool_tour[k]]  # 从四层嵌套降到二层
    #     num_intersections = check_intersections_between_tours(adjusted_sub_tours) 
    #     all_num_intersections[k] = num_intersections
    # print(f"  all_num_intersections: {all_num_intersections}") 

    #计算每个批次每个机器人的(总的转移距离与总的作业距离)和       
    for k in range(data.shape[0]):
        # print(f"Batch {k}:")
        for a in range(n_agent):
            total_trans_operate_distance[k][a]=subtour_trans_lengths[k][a]+sub_operate_distance[k][a]

     #计算每个批次每个机器人的(总的转移时间与总的作业时间)和       
    for k in range(data.shape[0]):
        # print(f"Batch {k}:")
        for a in range(n_agent):
            total_trans_operate_time[k][a]=subtour_trans_time[k][a]+sub_operate_time[k][a]
     # 加权奖励计算
    lambda_val = 0.5  # 拉格朗日乘数 
            
    # # 找出每个批次中，总距离和最大的那个机器人（转移距离+作业距离）
    # max_distance_agent_index = [-1 for _ in range(data.shape[0])]  # 存储每个批次中最大距离和对应的机器人索引
    # for k in range(data.shape[0]):
    #     max_sub_time = -1
    #     for a in range(n_agent):
    #         if total_trans_operate_time[k][a] > max_sub_time:
    #             max_sub_time = total_trans_operate_time[k][a]
    #             max_distance_agent_index[k] = a  # 更新最大距离和对应的机器人索引
    
    # # 打印每个批次中总距离和最大的那个机器人的工作量
    # for k in range(data.shape[0]):
    #     agent_id = max_distance_agent_index[k]  # 获取最大距离和对应的机器人索引
    #     if agent_id != -1:  # 确保找到了对应的机器人
    #         print(f"Batch {k}, Agent {agent_id} has the maximum total time (transfer+operate) with work load: {sub_operate_distance[k][agent_id]}")
    #         print(f"Batch {k}, Agent {agent_id} has the maximum total time (transfer+operate) with every operate load: {sub_every_operate[k][agent_id]}")
    #         print(f"Batch {k}, Agent {agent_id} has the maximum total time (transfer+operate) with trans load: {subtour_trans_lengths[k][agent_id]}")
    #     else:
    #         print(f"Batch {k} does not have a valid agent with maximum total distance.")
    #     #找出每个批次中，总距离和最大的那个机器人（转移距离+作业距离），输出 这个距离和 作为损失函数


    for k in range(data.shape[0]):
            # sub_max_trans_operate_distance[k]= max(total_trans_operate_distance[k])+ lambda_val * all_num_intersections[k]
            # sub_max_trans_operate_distance[k]= max(total_trans_operate_distance[k])-min(total_trans_operate_distance[k])
            # sub_max_trans_operate_distance[k]=max(sub_operate_distance[k])-min(sub_operate_distance[k])
        sub_max_trans_operate_distance[k]= max(total_trans_operate_distance[k])
        # sub_max_trans_distance[k] = max(total_trans_operate_distance[k])+lambda_val*(sum_difference_operate_distances[k])
        sub_max_trans_operate_time[k]=max(total_trans_operate_time[k])
        # sub_max_trans_distance[k] = max(subtour_trans_lengths[k])
        sub_max_trans_distance[k] = max(subtour_trans_lengths[k])+lambda_val*(sum_difference_operate_distances[k])
        
    last_reward = sub_max_trans_distance
    # last_reward=[-x for x in sub_max_trans_distance]
    # last_reward=sub_max_trans_operate_distance
    # for k in range(data.shape[0]):
    #     last_reward[k]= max(total_trans_operate_distance[k])
    #     # last_reward[k]=max(sub_operate_distance[k])
    
    return last_reward,sub_max_trans_operate_time,sub_max_trans_operate_distance



if __name__ == '__main__':
    from torch_geometric.data import Data
    from torch_geometric.data import Batch
    dev = 'cpu'
    torch.manual_seed(2)
    epsilon=0.1
    #准备数据-----------------------------------------------------------------
    # 解析数据文件 kroA100 0Farmland392 0Farmland81
    #berlin52  0Farmland  pcb442 eil101 pr76 lin105 ch130 tsp225 pcb1173 pr2392 ch150
    coordinates = parse_tsp('edata/random/200.tsp')
    workloads = parse_tour('edata/random/200.opt.tour')
    # print(coordinates)
    # 组合坐标和工作量
    fea_test = [coord + [workloads[i]] for i, coord in enumerate(coordinates)]
    
    # 假设你已经有了fea_test数据
    fea_test_normalized = normalize_data(fea_test)
    # print(fea_test_normalized)
    n_batch = 3
    n_agent = 3
    n_nodes = len(coordinates)
    # 生成随机数据zzzzzzzzzzzzzzzz
    # fea = torch.randint( size=[n_batch, n_nodes, 3]).float()
    fea = torch.rand(size=[n_batch, n_nodes, 3]) 
    # 将fea_test添加到fea的第一个batch中
    fea_get_r = fea.clone()
    fea[0] = torch.tensor(fea_test_normalized)#测试数据归一化后进入模型
    fea_get_r[0]=torch.tensor(fea_test)#根据分配方案，使用原始数据获得reward
    # print(f"打印实验数据fea_new的第一组数据:{fea_new[0]}")
    #----------------------------------------------------------------------
    

   

    adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])    
    data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(fea.shape[0])]
    # print(f"data_list:{data_list}")
    # generate batch graph
    batch_graph = Batch.from_data_list(data_list=data_list).to(dev)
    
    # test policy
    policy = Policy(in_chnl=fea.shape[-1], hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    path = './saved_model/{}.pth'.format(str(52) + '_3_gnn_tans_work_lam_3')
    policy.load_state_dict(torch.load(path, map_location=torch.device(dev)))
    pi = policy(batch_graph, n_nodes, n_batch)
    #print(f"pi:{pi}")
    # grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])
    
    # action, log_prob = action_argmax(pi,epsilon)
    action, log_prob = action_sample(pi)
   
    # optimized_actions = balance_workload(action,fea, n_agent)
    rewards,sub_max_trans_operate_time,sub_max_trans_operate_distance= get_reward(action, fea_get_r, n_agent)
    # loss = torch.mul(torch.tensor(rewards, device=dev), log_prob.sum(dim=1)).sum()
    # print(f"log_prob.sum(dim=1):{log_prob.sum(dim=1)}")
    # print(f"torch.tensor(rewards, device=dev):{torch.tensor(rewards, device=dev)}")
    
    print(f"action:{action}")
    # print(f"fea_new[0]:{fea_new[0]}")
    print(f"rewards:{rewards}")
    print(f"sub_max_trans_operate_time:{sub_max_trans_operate_time}")
    print(f"sub_max_trans_operate_distance:{sub_max_trans_operate_distance}")


# if __name__ == '__main__':
#     from torch_geometric.data import Data
#     from torch_geometric.data import Batch

#     dev = 'cpu'
#     torch.manual_seed(2)

#     n_agent = 4
#     n_nodes = 50
#     n_batch = 3
#     # get batch graphs data list
#     fea = torch.randint(low=10, high=1000, size=[n_batch, n_nodes, 2]).to(torch.float)  # [batch, nodes, fea]
#     adj = torch.ones([fea.shape[0], fea.shape[1], fea.shape[1]])
#     data_list = [Data(x=fea[i], edge_index=torch.nonzero(adj[i]).t()) for i in range(fea.shape[0])]
#     # generate batch graph
#     batch_graph = Batch.from_data_list(data_list=data_list).to(dev)

#     # test policy
#     policy = Policy(in_chnl=fea.shape[-1], hid_chnl=32, n_agent=n_agent, key_size_embd=64,
#                     key_size_policy=64, val_size=64, clipping=10, dev=dev)

#     pi = policy(batch_graph, n_nodes, n_batch)

#     grad = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])

#     action, log_prob = action_sample(pi)
#     print(action)
#     print(f"log_prob:{log_prob}")

#     rewards = get_reward(action, fea, n_agent)

#     print(f"rewards:{rewards}")


    # print(fea)
    # sub_tours = [[fea[b, 0]] for b in range(fea.shape[0])]
    # fea_repeat = fea.repeat(1, n_agent, 1).reshape(n_batch, n_agent, n_nodes, -1)
    # print(fea_repeat)
    # action_repeat = action + torch.arange(0, n_agent, n_agent*n_batch)
    # print(torch.arange(0, n_agent, n_agent*n_batch))
    # index_ops = (torch.arange(0, 3, 6)[:, None] + torch.arange(3)).view(-1)
    # print(index_ops)



    # grad1 = torch.autograd.grad(pi.sum(), [param for param in embd_net.parameters()])
    # print(grad1)
    # grad2 = torch.autograd.grad(pi.sum(), [param for param in policy.parameters()])
    # print(grad2)