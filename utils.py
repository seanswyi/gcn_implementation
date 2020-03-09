import pdb

import numpy as np
import torch


def accuracy(pred, labels):
    pred_idxs = torch.argmax(pred, dim=1)
    num_correct = (pred_idxs == labels).sum().item()
    acc = num_correct / len(labels)

    return acc


def create_adj_hat(adj_mat):
    adj_tilde = adj_mat + np.eye(N=adj_mat.shape[0])
    deg_tilde = create_deg_mat(adj_tilde)

    adj_hat = np.matmul(np.matmul(deg_tilde, adj_tilde), deg_tilde)

    return adj_hat


def create_adj_mat(nodes, edges):
    adj_mat = np.zeros(shape=(nodes.shape[0], nodes.shape[0]))

    for _, pair in enumerate(edges):
        adj_mat[pair[0], pair[1]] = 1

    return adj_mat


def create_deg_mat(adj_mat):
    deg_diag = adj_mat.sum(1)
    deg_inv = np.power(deg_diag, -1)
    deg_sqrt = np.sqrt(deg_inv)
    deg_tilde = np.diag(deg_sqrt)

    return deg_tilde


def get_optimizer(config, model):
    params = model.parameters()
    lr = config.lr

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=params, lr=lr)

    return optimizer


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
