from gnn_dispn_policy import Policy, action_sample, get_reward
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from gnn_dispn_validation import validate
from common import append_loss_to_file,append_reward0_to_file,append_valited_result_to_file,parse_tsp,parse_tour
import numpy as np


def train(batch_size, no_nodes, policy_net, l_r, no_agent, iterations, device):

    # prepare validation data
    validation_data = torch.load('./validation_data/validation_data5_'+str(no_nodes)+'_'+str(batch_size)+'0')
    # a large start point
    best_so_far = np.inf
    validation_results = []

    # optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=l_r)

    for itr in range(iterations):
        
        # prepare training data
        data = torch.rand(size=[batch_size, no_nodes, 3])  # [batch, nodes, fea], fea is 2D location
        # data = torch.randint(low=10, high=1000, size=[batch_size, n_nodes, 3]).float()
        # data[0] = torch.tensor(fea_test)
        adj = torch.ones([data.shape[0], data.shape[1], data.shape[1]])  # adjacent matrix fully connected
        data_list = [Data(x=data[i], edge_index=torch.nonzero(adj[i], as_tuple=False).t()) for i in range(data.shape[0])]
        batch_graph = Batch.from_data_list(data_list=data_list).to(device)

        # get pi
        pi = policy_net(batch_graph, n_nodes=data.shape[1], n_batch=batch_size)
        # sample action and calculate log probabilities
        action, log_prob = action_sample(pi)
        # append_loss_to_file(action[0])
        # append_loss_to_file(action[1])
        # append_loss_to_file(action[2])
        # append_loss_to_file(action[3])
        # append_loss_to_file(action[4])
        # append_loss_to_file(action[5])
        # get reward for each batch
        reward,_,_ = get_reward(action, data, no_agent)  # reward: tensor [batch, 1]
        # print('reward:'+str(reward))
        
        # compute loss
        loss = torch.mul(torch.tensor(reward, device=device) - 2, log_prob.sum(dim=1)).sum()
        print('loss:'+str(loss))
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sumreward = sum(reward)
        append_loss_to_file(loss)
        # if itr % 100 == 0:
        print('\nIteration:', itr)
        # print(format(sum(reward) / batch_size, '.4f'))
        # print(f"sum(reward)/batch_size-----:{sum(reward) / batch_size}")
        # print(f"sum(reward)-----:{sumreward}")
        # print(f"batch_size-----:{batch_size}")
        append_reward0_to_file(sumreward/batch_size)
        # validate and save best nets
        if (itr+1) % 5 == 0:
            validation_result = validate(validation_data, policy_net, no_agent, device)
            append_valited_result_to_file(validation_result)
            # print(f"validation_result:{validation_result}")
            if validation_result < best_so_far:
                torch.save(policy_net.state_dict(), './saved_model/{}_{}_gnn_tans_work_lam_3.pth'.format(str(no_nodes), str(no_agent)))
                print('Found better policy, and the validation result is:', format(validation_result, '.4f'))
                validation_results.append(validation_result)
                best_so_far = validation_result
    return validation_results




if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2)

    n_agent = 3
    n_nodes = 52
    batch_size = 512
    lr = 1e-4
    iteration = 20000

    

    policy = Policy(in_chnl=3, hid_chnl=32, n_agent=n_agent, key_size_embd=64,
                    key_size_policy=64, val_size=64, clipping=10, dev=dev)

    best_results = train(batch_size, n_nodes, policy, lr, n_agent, iteration, dev)
    print(min(best_results))
