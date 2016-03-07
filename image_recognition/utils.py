import tarfile
import os
import sys
import urllib.request
from hashlib import sha1

import tensorflow as tf
import numpy as np

from mnist import input_data

CACHE_DIR = "cache"
MODEL_DIR = "models"
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
GRAPH_DEF_PB = 'classify_image_graph_def.pb'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
GROUND_TRUTH_TENSOR_NAME = 'ground_truth:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


def hash_str(string, length=20):
    return sha1(string.encode()).hexdigest()[:length]


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def maybe_download_inception_and_extract():
    filename = DATA_URL.split('/')[-1]
    dir_name = os.path.join(MODEL_DIR, filename)
    ensure_dir_exists(dir_name)
    file_path = os.path.join(dir_name, filename)
    if not os.path.exists(file_path):
        def _progress(count, block_size, total_size):
            print('\r>> Downloading {} {:.1f}%'
                  .format(filename, float(count * block_size) / float(total_size) * 100.0), end="")

        file_path, _ = urllib.request.urlretrieve(DATA_URL, file_path, reporthook=_progress)

        print()
        statinfo = os.stat(file_path)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

        tarfile.open(file_path, 'r:gz').extractall(dir_name)
    return dir_name


def load_google_inception_graph(dir_name):
    with tf.Session() as sess:
        model_filename = os.path.join(dir_name, GRAPH_DEF_PB)
        with tf.gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    return sess.graph


def add_final_training_ops(graph, class_count, learning_rate=0.01, optimizer=tf.train.GradientDescentOptimizer):
    bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
    layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
                                name='final_weights')
    layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    keep_prob = tf.placeholder("float")
    bottleneck_tensor = tf.nn.dropout(bottleneck_tensor, keep_prob)

    logits = tf.matmul(bottleneck_tensor, layer_weights, name='final_matmul') + layer_biases
    softmax = tf.nn.softmax(logits, name='final_result')
    ground_truth_placeholder = tf.placeholder(tf.float32, [None, class_count], name=GROUND_TRUTH_TENSOR_NAME.split(":")[0])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, ground_truth_placeholder)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = optimizer(learning_rate).minimize(cross_entropy_mean)

    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(ground_truth_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return train_step, cross_entropy_mean, accuracy, keep_prob


def compute_bottlenecks(sess, graph, data, identificators, feed_placeholder, bottleneck_tensor, desc="data"):
    bottlenecks_values = []
    cache_dir = os.path.join(CACHE_DIR, "bottlenecks")
    ensure_dir_exists(cache_dir)
    iterations = len(data)
    computed = False
    for i, (image, name) in enumerate(zip(data, identificators)):
        cache_filename = os.path.join(cache_dir, "{}.npy".format(name))
        if os.path.exists(cache_filename):
            bottleneck_values = np.load(cache_filename)
        else:
            computed = True
            print('\r>> Computing bottlenecks for {}: {} from {} - {:.1f}%'
              .format(desc, i + 1, iterations, (i + 1) / iterations * 100.0), end="")
            bottleneck_values = sess.run(bottleneck_tensor, feed_dict={feed_placeholder: image})[0]
            np.save(cache_filename, bottleneck_values)
        bottlenecks_values.append(bottleneck_values)
    if computed:
        print()
    return np.array(bottlenecks_values)


def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
