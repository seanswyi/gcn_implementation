import pdb

from config import get_args
from dataset import Data
from models import GCN


def main():
    config = get_args()
    dataset = Data(config)

    adj_hat = dataset.adj_hat
    features= dataset.features
    labels = dataset.labels

    model = GCN(config=config, num_features=features.shape[1], num_classes=labels.shape[1])

    pdb.set_trace()


if __name__ == '__main__':
    main()
