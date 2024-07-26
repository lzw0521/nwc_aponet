import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class NetGAT(torch.nn.Module):
    def __init__(self, in_chnl, hid_chnl, num_heads=1):
        super(NetGAT, self).__init__()
        
        # GAT conv layers
        self.conv1 = GATConv(in_chnl, hid_chnl, heads=num_heads, concat=True)
        self.bn1 = torch.nn.BatchNorm1d(hid_chnl * num_heads)
        self.conv2 = GATConv(hid_chnl * num_heads, hid_chnl, heads=num_heads, concat=True)
        self.bn2 = torch.nn.BatchNorm1d(hid_chnl * num_heads)
        self.conv3 = GATConv(hid_chnl * num_heads, hid_chnl, heads=num_heads, concat=True)
        self.bn3 = torch.nn.BatchNorm1d(hid_chnl * num_heads)

        # Linears for distance feature
        self.distance_fc = nn.Linear(1, hid_chnl * num_heads)

        # Layers used in graph pooling
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(1+3):  # 1+x: 1 projection layer + x GAT layers
            self.linears_prediction.append(nn.Linear(hid_chnl * num_heads, hid_chnl * num_heads))

    def forward(self, x, edge_index, batch):
              
        # # Combine original node features with distance features
        # x = torch.cat([x, distance_features], dim=-1)
         # GAT层的前向传播
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # 全局池化
        gPool = global_mean_pool(x, batch)

        return x, gPool

# class NetGAT(torch.nn.Module):
#     def __init__(self, in_chnl, hid_chnl, num_heads=1):
#         super(NetGAT, self).__init__()

#         # GAT conv layers
#         self.conv1 = GATConv(in_chnl, hid_chnl, heads=num_heads, concat=True)
#         self.bn1 = torch.nn.BatchNorm1d(hid_chnl * num_heads)
#         self.conv2 = GATConv(hid_chnl * num_heads, hid_chnl, heads=num_heads, concat=True)
#         self.bn2 = torch.nn.BatchNorm1d(hid_chnl * num_heads)
#         self.conv3 = GATConv(hid_chnl * num_heads, hid_chnl, heads=num_heads, concat=True)
#         self.bn3 = torch.nn.BatchNorm1d(hid_chnl * num_heads)

#         # Layers used in graph pooling
#         self.linears_prediction = torch.nn.ModuleList()
#         for layer in range(1+3):  # 1+x: 1 projection layer + x GAT layers
#             self.linears_prediction.append(nn.Linear(hid_chnl * num_heads, hid_chnl * num_heads))

#     def forward(self, x, edge_index, batch):
#          # GAT层的前向传播
#         x = F.relu(self.bn1(self.conv1(x, edge_index)))
#         x = F.relu(self.bn2(self.conv2(x, edge_index)))
#         x = F.relu(self.bn3(self.conv3(x, edge_index)))

#         # 全局池化
#         gPool = global_mean_pool(x, batch)

#         return x, gPool


