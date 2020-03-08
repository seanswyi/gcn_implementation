import pdb

import torch.nn as nn

from utils import get_optimizer


class Trainer():
    def __init__(self, config, model, dataset):
        self.config = config
        self.num_epochs = self.config.num_epochs
        self.model = model
        self.dataset = dataset

        self.features = self.dataset.features
        self.adj_hat = self.dataset.adj_hat
        self.labels = self.dataset.labels
        self.idx_train = self.dataset.idx_train
        self.idx_valid = self.dataset.idx_valid
        self.idx_test = self.dataset.idx_test

    def train(self):
        self.model.train()

        optimizer = get_optimizer(self.config, self.model)
        loss_train = nn.NLLLoss()
        loss_valid = nn.NLLLoss()

        for epoch in range(self.num_epochs):
            pdb.set_trace()
            optimizer.zero_grad()
            output = self.model(self.features, self.adj_hat)

            output_train = output[self.idx_train]
            labels_train = self.labels[self.idx_train]
            output_valid = output[self.idx_valid]
            labels_valid = self.labels[self.idx_valid]

            loss_train = loss_train(output_train, labels_train)
            loss_train.backward()
            optimizer.step()

            loss_valid = loss_valid(output_valid, labels_valid)
