from config import get_args
from dataset import Data


def main():
    config = get_args()
    dataset = Data(config)
