from typing import Tuple, Union
import torch
from torch import nn
from torch import Tensor
from torch.functional import F
from torch_geometric.utils import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor

class GaussianExpansion(torch.nn.Module):
    def __init__(self, start: float, end: float, num_centers: int, sigma=0.5):
        super(GaussianExpansion, self).__init__()
        self.sigma = sigma
        self.start = start
        self.end = end
        self.num_centers = num_centers
        self.register_buffer('centers', torch.linspace(start, end, num_centers))

    def forward(self, edge_attr):
        device = self.centers.device
        edge_attr = edge_attr.view(-1, 1).to(device)
        rbf = torch.exp(-((edge_attr - self.centers) ** 2) / (self.sigma**2)).to(device)
        return rbf

    
class MEGNet_Edge(nn.Module):
    def __init__(self, dim=32):  
        super(MEGNet_Edge, self).__init__()
        self.edge_dense = nn.Sequential(nn.Linear(dim*4, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2)
                                        )

    def forward(self, src, dst, edge_attr, state, batch):
        comb = torch.concat([src, dst, edge_attr, state[batch]], dim=1)
        return self.edge_dense(comb)

class MEGNet_Node(nn.Module):
    def __init__(self, dim=32):
        super(MEGNet_Node, self).__init__()
        self.node_dense = nn.Sequential(nn.Linear(dim*3, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2)
                                        )

    def forward(self, x, edge_index, edge_attr, state, batch):
        v_mean = scatter(src=edge_attr,
                         index=edge_index[0,:],
                         dim=0,
                         dim_size=x.size(0),
                         reduce='mean')
        comb = torch.concat([x, v_mean, state[batch]], dim=1)
        return self.node_dense(comb)

class MEGNet_State(nn.Module):
    def __init__(self, dim=32):
        super(MEGNet_State, self).__init__()
        self.state_dense = nn.Sequential(nn.Linear(dim*3, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Dropout(0.2)
                                         )

    def forward(self, x, edge_index, edge_attr, state, batch):
        edge_batch_map = batch[edge_index[0]]
        u_e = scatter(src=edge_attr,
                      index=edge_batch_map,
                      dim=0,
                      dim_size=state.size(0),
                      reduce='mean')
        u_v = scatter(src=x,
                      index=batch,
                      dim=0,
                      dim_size=state.size(0),
                      reduce='mean')
        comb = torch.concat([u_e, u_v, state], dim=1)
        return self.state_dense(comb)
    
class CGCNNBlock(MessagePassing):
    def __init__(self, n_node_features, n_edge_features, aggr='add', batch_norm=True):
        super(CGCNNBlock, self).__init__(aggr=aggr)
        self.fc1 = nn.Linear(2*n_node_features+n_edge_features, n_node_features)
        self.fc2 = nn.Linear(2*n_node_features+n_edge_features, n_node_features)
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_node_features)
        else:
            self.bn = None

    def forward(self, x, edge_index, edge_attr):
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out if self.bn is None else self.bn(out)
        out = out + x[1]
        return out

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        return F.sigmoid(self.fc1(z)) * F.softplus(self.fc2(z))