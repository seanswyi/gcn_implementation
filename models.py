import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mat = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x, adj_mat):
        """
        x: N x C
        adj_mat: N x N
        """
        weight_prod = torch.DoubleTensor(self.weight_mat(x))
        output = torch.matmul(adj_mat, weight_prod)

        return output


class GCN(nn.Module):
    def __init__(self, config, num_features, num_classes):
        super().__init__()
        self.config = config
        self.num_hidden = self.config.num_hidden
        self.num_classes = num_classes
        self.num_features = num_features
        self.p = self.config.dropout_rate

        self.gc1 = GraphConv(in_features=self.num_features, out_features=self.num_hidden)
        self.gc2 = GraphConv(in_features=self.num_hidden, out_features=self.num_classes)

    def forward(self, x, adj_hat):
        x = F.relu(self.gc1(x, adj_hat))
        x = F.dropout(input=x, p=self.p, training=self.training)
        output = F.softmax(self.gc2(x, adj_hat), dim=1)

        return output
