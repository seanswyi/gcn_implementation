import numpy as np


def one_hot_enc(labels):
    """
    Simple one hot encoder for labels.

    We first construct a dictionary that maps labels to unique indices.
    We then construct a template of zeros and change the appropriate indices
      to 1.
    """
    classes = {label: idx for idx, label in enumerate(set(labels))}
    template = np.zeros(shape=(labels.shape[0], len(classes)))

    for idx, label in enumerate(labels):
        template[idx][classes[label]] = 1

    return template


def create_adj_mat(nodes, edges):
    adj_mat = np.zeros(shape=(nodes.shape[0], nodes.shape[0]))

    for _, pair in enumerate(edges):
        adj_mat[pair[0], pair[1]] = 1

    return adj_mat


def create_deg_mat(adj_mat):
    deg_mat = np.zeros(shape=adj_mat.shape)

    for idx, row in enumerate(adj_mat):
        deg_mat[idx, idx] = row.sum()

    return deg_mat
