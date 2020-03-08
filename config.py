import argparse


def get_args():
    argp = argparse.ArgumentParser(description='GCN',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data stuff.
    argp.add_argument('--data_dir', type=str, default='./data/')

    # Model stuff.
    argp.add_argument('--num_hidden', type=int, default=16)
    argp.add_argument('--dropout_rate', type=float, default=0.5)

    # Model running stuff.
    argp.add_argument('--num_epochs', type=int, default=200)
    argp.add_argument('--lr', type=float, default=0.01)
    argp.add_argument('--optimizer', type=str, default='adam')

    return argp.parse_args()
