from policy import Policy, action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from validation import validate
import numpy as np
from common import append_loss_to_file



def train(batch_size, no_nodes, policy_net, l_r, no_agent, iterations, device):

    # prepare validation data
    validation_data = torch.load('./validation_data/validation_data_'+str(no_nodes)+'_'+str(batch_size)+'0')
     # 打印第一批数据和长度
    print("训练验证数据的第一批数据:", validation_data[0])
    print("数据长度:", len(validation_data[0]))
    # 打印批次数量
    print("批次数量:", validation_data.size(0))
    total_distances = [0 for _ in range(validation_data.shape[0])]
    # a large start point
    best_so_far = np.inf
    validation_results = []
    
    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    for itr in range(iterations):
        print('\nIteration:', itr)
        # prepare training data
        data = torch.rand(size=[batch_size, no_nodes, 3])  # [batch, nodes, fea], fea is 2D location
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)
        
        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=batch_size)
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        # get reward for each batch
        reward,_,_ = get_reward(action, data, no_agent)  # reward: tensor [batch, 1]
        #print('reward:'+str(reward))
        # compute loss
        loss = torch.mul(torch.tensor(reward, device=device), log_prob.sum(dim=1)).sum()
        
        print(f"loss is:{loss}")
        append_loss_to_file(loss)
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"sum(reward)/batch_size-----:{sum(reward) / batch_size}")
        
        # validate and save best nets
        if (itr+1) % 5 == 0:
            validation_result,total_distances,reward = validate(validation_data, policy_net, no_agent, device)
            
            print(f"reward[0]:{reward}")
            print(f"total_distances[0]:{total_distances}")
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), './saved_model/{}_{}7.pth'.format(str(no_nodes), str(no_agent)))
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
                print('sum(reward)/batch_size,validation_results:'+str(validation_results))
                print(f"total_distances[0]:{total_distances}")
    return validation_results


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    n_agent = 5
    n_nodes = 52
    batch_size = 512
    lr = 1e-3
    iteration = 100

    policy = Policy(in_chnl=3, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)
    
    best_results = train(batch_size, n_nodes, policy, lr, n_agent, iteration, dev)
    print(min(best_results))
