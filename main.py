import torch

from config import get_args
from dataset import Data
from models import GCN
from solver import Solver


def main():
    config = get_args()
    dataset = Data(config)

    num_features = dataset.features.shape[1]
    num_classes = dataset.labels.max().item() + 1

    model = GCN(config=config, num_features=num_features, num_classes=num_classes)
    solver = Solver(config, model, dataset)

    if torch.cuda.is_available():
        model = model.to('cuda')

    criterion, best_model = solver.train()
    solver.test(criterion, best_model)


if __name__ == '__main__':
    main()
