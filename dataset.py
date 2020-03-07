import os
import pdb

import numpy as np
import torch

from utils import create_adj_mat, create_adj_hat, one_hot_enc


class Data():
    def __init__(self, config):
        self.config = config
        self.nodes_path = os.path.join(self.config.data_dir, 'cora.content.txt')
        self.edges_path = os.path.join(self.config.data_dir, 'cora.cites.txt')

        self.load()

    def load(self):
        nodes = np.genfromtxt(fname=self.nodes_path, dtype=np.dtype(str))
        edges = np.genfromtxt(fname=self.edges_path, dtype=np.int)

        idxs = np.array(nodes[:, 0], dtype=np.int)
        idxs_features = np.array(nodes[:, :-1], dtype=np.float32)
        labels = nodes[:, -1]

        idx_ordered = {og: new for new, og in enumerate(idxs)}

        for idx, row in enumerate(idxs_features):
            idxs_features[idx][0] = idx_ordered[row[0]]

        idxs = np.array(idxs_features[:, 0], dtype=np.int)
        features = np.array(idxs_features[:, 1:], dtype=np.float)
        labels = one_hot_enc(labels)

        for idx, pair in enumerate(edges):
            edges[idx] = [idx_ordered[pair[0]], idx_ordered[pair[1]]]

        adj_mat = create_adj_mat(nodes, edges)
        adj_hat = create_adj_hat(adj_mat)

        self.adj_hat = torch.tensor(adj_hat)
        self.features = torch.tensor(features)
        self.edges = torch.tensor(edges)
        self.labels = torch.tensor(np.where(labels)[1])

        self.idx_train = torch.tensor(range(140))
        self.idx_valid = torch.tensor(range(200, 500))
        self.idx_test = torch.tensor(range(500, 1500))

        if torch.cuda.is_available():
            self.adj_hat = self.adj_hat.to('cuda')
            self.features = self.features.to('cuda')
            self.edges = self.edges.to('cuda')
            self.labels = self.labels.to('cuda')
