import numpy as np


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
