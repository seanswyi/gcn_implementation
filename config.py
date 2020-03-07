import argparse


def get_args():
    argp = argparse.ArgumentParser(description='GCN',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data stuff.
    argp.add_argument('--data_dir', type=str, default='./data/')

    # Model stuff.
    argp.add_argument('--num_hidden', type=int, default=16)

    return argp.parse_args()
