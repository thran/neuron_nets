import os

import numpy as np
from hashlib import sha1
import tensorflow as tf


def hash_str(string, length=20):
    return sha1(string.encode()).hexdigest()[:length]


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def in_top_k(predictions, labels, k):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(predictions, labels, k)))


def dense_to_one_hot(labels, num_classes):
    nans = np.array([x is None for x in labels])
    labels[nans] = 0
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[list(index_offset + labels.ravel())] = 1
    labels_one_hot[nans, 0] = 0
    return labels_one_hot