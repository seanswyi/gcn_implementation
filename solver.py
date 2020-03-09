import pdb

import torch.nn as nn
import torch.nn.functional as F

from utils import accuracy, get_optimizer


class Solver():
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
        criterion = nn.NLLLoss()

        best_acc_valid = 0.0
        best_model = None

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            output = self.model(self.features, self.adj_hat)

            output_train = output[self.idx_train]
            labels_train = self.labels[self.idx_train]
            output_valid = output[self.idx_valid]
            labels_valid = self.labels[self.idx_valid]

            loss_train = criterion(output_train, labels_train)

            loss_train.backward()
            optimizer.step()

            loss_valid = criterion(output_valid, labels_valid).item()

            acc_train = accuracy(output_train, labels_train)
            acc_valid = accuracy(output_valid, labels_valid)

            if acc_valid > best_acc_valid:
                best_acc_valid = acc_valid
                best_model = self.model

            print('Epoch {}'.format(epoch + 1))
            print('\t Training loss: {} \t Validation loss: {}'.format(loss_train.item(), loss_valid))
            print('\t Training accuracy: {} \t Validation accuracy: {}'.format(acc_train, acc_valid))
            print()

        return criterion, best_model

    def test(self, criterion, model):
        model.eval()

        output = model(self.features, self.adj_hat)
        output_test = output[self.idx_test]
        labels_test = self.labels[self.idx_test]

        loss_test = criterion(output_test, labels_test)
        acc_test = accuracy(output_test, labels_test)

        print('=================================================')
        print('Testing loss: {} \t Testing accuracy: {}'.format(loss_test.item(), acc_test))
        print('=================================================')
