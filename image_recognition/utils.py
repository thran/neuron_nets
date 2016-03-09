import tarfile
import os
import urllib.request
from hashlib import sha1
import tensorflow as tf
import numpy as np

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
GRAPH_DEF_PB = 'classify_image_graph_def.pb'
MODEL_DIR = "models"


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


def compute_bottlenecks(sess, graph, data, identificators, feed_placeholder, bottleneck_tensor, cache_dir):
    bottlenecks_values = []
    cache_dir = os.path.join(cache_dir, "bottlenecks")
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
                  .format("data", i + 1, iterations, (i + 1) / iterations * 100.0), end="")
            bottleneck_values = sess.run(bottleneck_tensor, feed_dict={feed_placeholder: image})[0]
            np.save(cache_filename, bottleneck_values)
        bottlenecks_values.append(bottleneck_values)
    if computed:
        print()
    return np.array(bottlenecks_values)


def in_top_k(predictions, labels, k):
    return tf.reduce_mean(tf.to_float(tf.nn.in_top_k(predictions, labels, k)))


def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
