import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedded_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedded_dim)
        self.classifier = torch.nn.Linear(embedded_dim, num_classes)

    def encode(self, x, edge_index):
        x = x.to(torch.float)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(x)
        x = self.classifier(x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, subgraph_feat_dim=None):
        super().__init__()
        out_channels = 1
        self.subgraph_feat_dim = 0 if subgraph_feat_dim is None else subgraph_feat_dim
        self.lin1 = torch.nn.Linear(input_dim * 2 + self.subgraph_feat_dim, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        if self.subgraph_feat_dim:
            self.forward = lambda x_i, x_j, subgraph_features: self._forward(x_i, x_j, subgraph_features)
        else:
            self.forward = lambda x_i, x_j: self._forward(x_i, x_j)

    def _forward(self, x_i, x_j, subgraph_features=None):
        if self.subgraph_feat_dim:
            x = torch.cat([x_i, x_j, subgraph_features], dim=-1)
        else:
            x = torch.cat([x_i, x_j], dim=-1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
