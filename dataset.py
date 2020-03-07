import os

import numpy as np

from utils import one_hot_enc


class Data():
    def __init__(self, config):
        self.config = config
        self.nodes_path = os.path.join(self.config.data_dir, 'cora.content.txt')
        self.edges_path = os.path.join(self.config.data_dir, 'cora.cites.txt')

    def load(self):
        nodes = np.genfromtxt(fname=self.nodes_path, dtype=np.dtype(str))
        edges = np.genfromtxt(fname=self.edges_path, dtype=np.int)

        idxs = np.array(nodes[:, 0], dtype=np.int)
        idxs_features = nodes[:, :-1]
        labels = nodes[:, -1]

        idx_ordered = {og: new for new, og in enumerate(idxs)}

        for idx, row in enumerate(idxs_features):
            idxs_features[idx][0] = idx_ordered[row[0]]

        idxs = np.array(idxs_features[:, 0], dtype=np.int32)
        features = np.array(idxs_features[:, 1:], dtype=np.float32)
        labels = one_hot_enc(labels)

        for idx, pair in enumerate(edges):
            edges[idx] = [idx_ordered[pair[0]], idx_ordered[pair[1]]]

        adj_mat = self.create_adj(nodes, edges)

    def create_adj(self, nodes, edges):
        return None
