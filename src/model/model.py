from src.layers.layer import RBFExpansion, MEGNetBlock
from torch_geometric.nn import Set2Set, global_add_pool, global_max_pool, global_mean_pool, CGConv
from torch import nn
import torch

class MEGNETModel(nn.Module):
    def __init__(self, n_node_features, n_edge_features, n_state_features, hidden_dim=32):
        super(MEGNETModel, self).__init__()
        self.rbf = RBFExpansion(start=0.0, end=5.0, num_centers=100)
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=n_node_features)
        self.megnet_blocks1 = MEGNetBlock(n_node_features, n_edge_features, n_state_features)
        self.megnet_blocks2 = MEGNetBlock(n_node_features=hidden_dim, n_edge_features=hidden_dim, n_state_features=hidden_dim)
        self.set2set_nodes = Set2Set(in_channels=hidden_dim,
                                     processing_steps=3,
                                     num_layers=1)
        self.set2set_edges = Set2Set(in_channels=hidden_dim,
                                     processing_steps=3,
                                     num_layers=1)
        self.dense = nn.Sequential(
            nn.Linear(2*hidden_dim+2*hidden_dim+hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(16, 1) 

    def forward(self, data):
        x = self.embedding(data.x).squeeze()
        edge_attr = self.rbf(edge_attr=data.edge_attr, device=data.edge_attr.device)
        x, edge_attr, state = self.megnet_blocks1(x, data.edge_index, edge_attr, data.state, data.batch) 
        x, edge_attr, state = self.megnet_blocks2(x, data.edge_index, edge_attr, state, data.batch)
        x = self.set2set_nodes(x, data.batch)
        edge_attr = self.set2set_edges(edge_attr, data.batch[data.edge_index[0]], dim_size=state.size(0))
        z = torch.concat([x, edge_attr, state], dim=1)
        dense_output = self.dense(z) 
        out = self.output(dense_output)
        return out
    
class CGCNNModel(nn.Module):
    def __init__(self, n_node_features, n_edge_features, num_blocks):
        super(CGCNNModel, self).__init__()
        
        self.rbf = RBFExpansion(start=0.0, end=5.0, num_centers=100)
        self.CGCNNConv = nn.ModuleList([CGConv(n_node_features, n_edge_features, aggr='mean', batch_norm=True) for i in range(num_blocks)])
        self.dense = nn.Sequential(nn.Linear(3*n_node_features, 32),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(32, 16),
                                    nn.ReLU(),
                                    nn.Dropout(0.2))
        self.out = nn.Linear(16,1)

    def forward(self, data):
        x, edge_index, edge_attr, batch_index = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = self.rbf(edge_attr=edge_attr, device=edge_attr.device)
        for block in self.CGCNNConv:
            x = block(x, edge_index, edge_attr)
        x = torch.cat([global_mean_pool(x, batch_index),
                       global_max_pool(x, batch_index),
                       global_add_pool(x, batch_index)], dim=1)
        x = self.dense(x)
        out = self.out(x)
        return out