import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import torch

class PoolModel(nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Initial Graph Convolution
        self.conv1 = GCNConv(data.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Regression head to predict your CFD values (e.g., Ux, Uy, P)
        self.out = nn.Linear(hidden_channels, 5)

    def forward(self, data):
        # 1. Obtain node embeddings

        x, edge_index= data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Node-level output (for the full 50,480 vector prediction)
        return self.out(x)
#%%

class GNNMessagePassing(MessagePassing):
    def __init__(self, node_input, edge_in, hidden_channels):
        super(GNNMessagePassing, self).__init__(aggr='add')

        self.msg_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels + edge_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self,x_i, x_j,edge_attr):
        combined = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(combined)

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=-1)
        return self.upd_mlp(combined)
#%%
#using the message passing model

class FlowPredictor(torch.nn.Module):
    def __init__(self,hidden_channels):
        super().__init__()
        self.num_layers = 4
        self.layers = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.layers.append(GNNMessagePassing(node_input=hidden_channels, edge_in=3, hidden_channels=hidden_channels)) ## what sort of layers do we have in our module list.
        self.encoder = nn.Linear(5, hidden_channels) # [x, y, u, v, p] -> 64
#        self.processor = GNNMessagePassing(node_input=hidden_channels, edge_in=3, hidden_channels=hidden_channels)
        self.decoder = nn.Linear(hidden_channels, 5) # hidden_channels -> [u, v, p]

    def forward(self, data):
        h = self.encoder(data.x)
        for layer in self.layers:
            h_update = layer(h, data.edge_index, data.edge_attr)

        h = h + h_update
        return self.decoder(h)
