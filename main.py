import pdb

import torch

from config import get_args
from dataset import Data
from models import GCN
from solver import Trainer


def main():
    config = get_args()
    dataset = Data(config)

    num_features = dataset.features.shape[1]
    num_classes = dataset.labels.max().item() + 1

    model = GCN(config=config, num_features=num_features, num_classes=num_classes)
    trainer = Trainer(config, model, dataset)

    if torch.cuda.is_available():
        model = model.to('cuda')

    trainer.train()

    pdb.set_trace()


if __name__ == '__main__':
    main()
