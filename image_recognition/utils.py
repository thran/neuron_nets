import tarfile
import os
import urllib.request
from hashlib import sha1
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape

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


def compute_bottlenecks(sess, data, identificators, feed_placeholder, bottleneck_tensor,
                        cache_dir=None, prepare_function=None):
    bottlenecks_values = []
    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, "bottlenecks")
        ensure_dir_exists(cache_dir)
    iterations = len(data)
    computed = False
    for i, (image, name) in enumerate(zip(data, identificators)):
        if cache_dir is None:
            bottleneck_values = sess.run(bottleneck_tensor, feed_dict={feed_placeholder: image})[0]
        else:
            cache_filename = os.path.join(cache_dir, "{}.npy".format(name))
            if os.path.exists(cache_filename):
                bottleneck_values = np.load(cache_filename)
            else:
                computed = True
                print('\r>> Computing bottlenecks for {}: {} from {} - {:.1f}%'
                      .format("data", i + 1, iterations, (i + 1) / iterations * 100.0), end="")
                if prepare_function:
                    image = prepare_function(sess, [image])[0]
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


def add_input_distortions(image_data_placeholder, input_height, input_width, input_depth,
                          flip_left_right=False, max_crop=1, min_crop=1, random_brightness=1):
    decoded_image = tf.image.decode_jpeg(image_data_placeholder)
    image = tf.cast(decoded_image, dtype=tf.float32)
    image_4d = tf.expand_dims(image, 0)

    scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1 / max_crop, maxval=1 / min_crop)

    precrop_width, precrop_height = tf.mul(scale_value, input_width), tf.mul(scale_value, input_height)
    precrop_shape_shape = tf.cast(tf.pack([precrop_height, precrop_width]), dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(image_4d, precrop_shape_shape)
    image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(image_3d, [input_height, input_width, input_depth])

    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - random_brightness
    brightness_max = 1.0 + random_brightness
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.mul(flipped_image, brightness_value)
    return tf.expand_dims(brightened_image, 0)


def represent(obj):
    if type(obj) is not dict:
        return str(obj)
    parts = []
    for key in sorted(obj.keys()):
        parts.append("'{}': {}".format(key, represent(obj[key])))
    return "{" + ", ".join(parts) + "}"
