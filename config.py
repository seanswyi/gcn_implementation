import argparse


def get_args():
    argp = argparse.ArgumentParser(description='GCN',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data stuff.
    argp.add_argument('--data_dir', type=str, default='../graph_data')

    return argp.parse_args()
