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
        edge_attr = edge_attr.view(-1, 1).to(device)  # Shape (num_edges, 1)
        rbf = torch.exp(-((edge_attr - self.centers) ** 2) / (self.sigma**2)).to(device)
        return rbf


class MEGNet_Edge(nn.Module):
    def __init__(self, dim=32):  
        super(MEGNet_Edge, self).__init__()
        self.edge_dense = nn.Sequential(nn.Linear(dim*4, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Linear(dim, dim),
                                        nn.BatchNorm1d(dim))

    def forward(self, x, edge_index, edge_attr, state, batch):
        src, dst = edge_index
        edge_batch_map = batch[src]
        comb = torch.concat([x[src], x[dst], edge_attr, state[edge_batch_map]], dim=1)
        return self.edge_dense(comb)

class MEGNet_Node(nn.Module):
    def __init__(self, dim=32):
        super(MEGNet_Node, self).__init__()
        self.node_dense = nn.Sequential(nn.Linear(dim*3, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Linear(dim, dim),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(dim),
                                        nn.Linear(dim, dim),
                                        nn.BatchNorm1d(dim))

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
                                         nn.Linear(dim, dim),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(dim),
                                         nn.Linear(dim, dim),
                                         nn.BatchNorm1d(dim))

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
        
class MEGNetBlock(nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_state_features):
        super(MEGNetBlock, self).__init__()
        self.e_dense = nn.Sequential(nn.Linear(n_edge_features, 64),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(32))
        self.v_dense = nn.Sequential(nn.Linear(n_node_features, 64),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(32))
        self.u_dense = nn.Sequential(nn.Linear(n_state_features, 64),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(64),
                                     nn.Linear(64, 32),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(32))
        self.update_edge = MEGNet_Edge()
        self.update_node = MEGNet_Node()
        self.update_state = MEGNet_State()
        
    def forward(self, x, edge_index, edge_attr, state, batch):
        e = self.e_dense(edge_attr)
        v = self.v_dense(x)
        u = self.u_dense(state)
        e = self.update_edge(v, edge_index, e, u, batch) + e
        v = self.update_node(v, edge_index, e, u, batch) + v
        u = self.update_state(v, edge_index, e, u, batch) + u
        return v, e, u