from src.layers.layer import GaussianExpansion, MEGNetBlock, CGCNNBlock, MEGNet_Edge, MEGNet_Node, MEGNet_State
from torch_geometric.nn import Set2Set, global_add_pool, global_max_pool, global_mean_pool, MetaLayer
from torch import nn
import torch

class MEGNETModel(nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_state_features, hidden_dim=32, nblocks=3):
        super(MEGNETModel, self).__init__()
        self.nblocks = nblocks
        self.ge = GaussianExpansion(start=0.0, end=5.0, num_centers=100)
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=n_node_features)
        self.e_dense_list = nn.ModuleList()
        self.v_dense_list = nn.ModuleList()
        self.u_dense_list = nn.ModuleList()
        self.update_list = nn.ModuleList()
        for i in range(nblocks):
            if i == 0:
                e_dense = nn.Sequential(nn.Linear(n_edge_features, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
                v_dense = nn.Sequential(nn.Linear(n_node_features, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
                u_dense = nn.Sequential(nn.Linear(n_state_features, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
                self.e_dense_list.append(e_dense)
                self.v_dense_list.append(v_dense)
                self.u_dense_list.append(u_dense)
                self.update_list.append(
                    MetaLayer(
                        MEGNet_Edge(),
                        MEGNet_Node(),
                        MEGNet_State()
                    )
                )
            elif i > 0:
                e_dense = nn.Sequential(nn.Linear(hidden_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
                v_dense = nn.Sequential(nn.Linear(hidden_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
                u_dense = nn.Sequential(nn.Linear(hidden_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
                self.e_dense_list.append(e_dense)
                self.v_dense_list.append(v_dense)
                self.u_dense_list.append(u_dense)
                self.update_list.append(
                    MetaLayer(
                        MEGNet_Edge(),
                        MEGNet_Node(),
                        MEGNet_State()
                    )
                )
        self.set2set_nodes = Set2Set(in_channels=hidden_dim,
                                     processing_steps=3,
                                     num_layers=1)
        self.set2set_edges = Set2Set(in_channels=hidden_dim,
                                     processing_steps=3,
                                     num_layers=1)
        self.dense = nn.Sequential(
            nn.Linear(2*hidden_dim+2*hidden_dim+hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(16, 1) 

    def forward(self, data):
        x = self.embedding(data.x).squeeze()
        edge_attr = self.ge(edge_attr=data.edge_attr)
        for i in range(self.nblocks):
            if i == 0:
                e_dense = self.e_dense_list[i](edge_attr)
                v_dense = self.v_dense_list[i](x)
                u_dense = self.u_dense_list[i](data.state)
                v_out, e_out, u_out = self.update_list[i](v_dense, data.edge_index, e_dense, u_dense, data.batch)
                e = torch.add(e_out, e_dense)
                v = torch.add(v_out, v_dense)
                u = torch.add(u_out, u_dense)
            elif i > 0:
                e_dense = self.e_dense_list[i](e)
                v_dense = self.v_dense_list[i](v)
                u_dense = self.u_dense_list[i](u)
                v_out, e_out, u_out = self.update_list[i](v_dense, data.edge_index, e_dense, u_dense, data.batch)
                e = torch.add(e_out, e)
                v = torch.add(v_out, v)
                u = torch.add(u_out, u)
        v = self.set2set_nodes(v, data.batch)
        e = self.set2set_edges(e, data.batch[data.edge_index[0]], dim_size=u.size(0))
        z = torch.concat([v, e, u], dim=1)
        dense_output = self.dense(z) 
        out = self.output(dense_output)
        return out
    
class CGCNNModel(nn.Module):
    def __init__(self, n_node_features, n_edge_features, num_blocks):
        super(CGCNNModel, self).__init__()
        
        self.ge = GaussianExpansion(start=0.0, end=5.0, num_centers=100)
        self.CGCNNConv = nn.ModuleList([CGCNNBlock(n_node_features, n_edge_features, aggr='add', batch_norm=True) for i in range(num_blocks)])
        self.dense = nn.Sequential(nn.Linear(3*n_node_features, 32),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(32, 16),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        self.out = nn.Linear(16,1)

    def forward(self, data):
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self.ge(edge_attr=edge_attr)
        for block in self.CGCNNConv:
            x = block(x, edge_index, edge_attr)
        x = torch.cat([global_mean_pool(x, batch_index),
                       global_max_pool(x, batch_index),
                       global_add_pool(x, batch_index)], dim=1)
        x = self.dense(x)
        out = self.out(x)
        return out