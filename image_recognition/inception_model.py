import numpy as np
import os

import tensorflow as tf
from tensorflow.python.client.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile

import inception.inception_model as inception
from image_recognition.datasets.dataset_processor import download_flowerchecker_dataset
from image_recognition.utils import in_top_k, ensure_dir_exists, const_for_none
from inception import slim
from image_recognition.fc_datasets import FlowerCheckerDataSet

TENSOR_BOARD_DIR = "../tenzor_board"
ORIGINAL_INCEPTION_CKPT_DIR = "models/inception-v3"
CACHE_DIR = "/home/thran/projects/cache"
MODEL_DIR = "models"
INPUT_SIZE = 299, 299
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
INITIAL_LR = 0.001
LR_DECAY_FACTOR = 0.3
LR_DECAY_STEP = 10 * 10 ** 5 / 20   # epochs * images / batch_size


class InceptionModel:
    # >=0.2 with meta
    VERSION = '0.2'

    def __init__(self, dataset):
        self._data_set = dataset
        self.class_count = dataset.class_count
        self.jpeg = tf.placeholder(dtype='string', name="jpeg")

        lat_placeholder = tf.placeholder_with_default(tf.zeros([], dtype=tf.float32), [], name='lat_placeholder')
        self.lat_placeholder = tf.placeholder_with_default(tf.expand_dims(lat_placeholder, 0), [None])
        lng_placeholder = tf.placeholder_with_default(tf.zeros([], dtype=tf.float32), [], name='lng_placeholder')
        self.lng_placeholder = tf.placeholder_with_default(tf.expand_dims(lng_placeholder, 0), [None])
        week_placeholder = tf.placeholder_with_default(tf.zeros([], dtype=tf.float32), [], name='week_placeholder')
        self.week_placeholder = tf.placeholder_with_default(tf.expand_dims(week_placeholder, 0), [None])

        self.ground_truth = tf.placeholder(tf.float32, [None, self.class_count])

        self.processed_jpeq = None
        self.distorted_image = None
        self.inception_input = None
        self.logits = None
        self.predictions = None
        self.total_loss = None
        self.cross_entropy = None
        self.train_step = None
        self.accuracy = None
        self.top3 = None
        self.top5 = None

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(INITIAL_LR, self.global_step,
                                                        LR_DECAY_STEP, LR_DECAY_FACTOR, staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, RMSPROP_DECAY,
                                                   momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)
        self.tensor_board_path = os.path.join(TENSOR_BOARD_DIR, str(self))
        self.save_path = os.path.join(CACHE_DIR, "checkpoints", str(self))

    def __str__(self):
        return "IncMod v{} - {} plants".format(self.VERSION, self._data_set.class_count)

    def add_image_pre_processing(self):
        with tf.variable_scope('pre_process_image'):
            image = tf.image.decode_jpeg(self.jpeg, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.sub(image, 0.5)
            image = tf.mul(image, 2.0)
            self.processed_jpeq = tf.image.resize_images(image, *INPUT_SIZE)
            self.inception_input = tf.placeholder_with_default(
                tf.expand_dims(self.processed_jpeq, 0), shape=[None, INPUT_SIZE[0], INPUT_SIZE[1], 3])

    def pre_process_image(self, sess, image_path):
        with tf.gfile.FastGFile(image_path, 'rb') as image_file:
            image = sess.run(self.processed_jpeq, feed_dict={self.jpeg: image_file.read()})
        return image

    def add_image_distortion(self):
        with tf.variable_scope('distort_image'):
            image = tf.image.decode_jpeg(self.jpeg, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            crop_scale = tf.random_uniform([], minval=0.5, maxval=1)
            height = tf.cast(INPUT_SIZE[0] / crop_scale, tf.int32)
            width = tf.cast(INPUT_SIZE[1] / crop_scale, tf.int32)
            image = tf.image.resize_images(image, height, width)

            image = tf.random_crop(image, [INPUT_SIZE[0], INPUT_SIZE[1], 3])
            image = tf.image.random_flip_left_right(image)

            def distort_colors_1():
                i = tf.image.random_brightness(image, max_delta=32. / 255.)
                i = tf.image.random_saturation(i, lower=0.5, upper=1.5)
                i = tf.image.random_hue(i, max_delta=0.2)
                i = tf.image.random_contrast(i, lower=0.5, upper=1.5)
                return i

            def distort_colors_2():
                i = tf.image.random_brightness(image, max_delta=32. / 255.)
                i = tf.image.random_contrast(i, lower=0.5, upper=1.5)
                i = tf.image.random_saturation(i, lower=0.5, upper=1.5)
                i = tf.image.random_hue(i, max_delta=0.2)
                return i

            image = tf.cond(tf.equal(0, tf.random_uniform(shape=[], maxval=2, dtype=tf.int32)),
                            distort_colors_1, distort_colors_2)

            image = tf.sub(image, 0.5)
            image = tf.mul(image, 2.0)
            self.distorted_image = image

    def distort_image(self, sess, image_path):
        with tf.gfile.FastGFile(image_path, 'rb') as image_file:
            image = sess.run(self.distorted_image, feed_dict={self.jpeg: image_file.read()})
        return image

    def add_train_step(self):
        with tf.variable_scope('taining'):
            loss = slim.losses.cross_entropy_loss(self.logits[0], self.ground_truth, label_smoothing=0.1, weight=1.0)
            loss_auxiliary = slim.losses.cross_entropy_loss(self.logits[1], self.ground_truth, label_smoothing=0.1, weight=0.4, scope='aux_loss')
            losses = [loss, loss_auxiliary]
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(losses + [total_loss])

            with tf.control_dependencies([loss_averages_op]):
                self.total_loss = tf.identity(total_loss)

            apply_gradient_op = self.optimizer.minimize(self.total_loss)

            variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, num_updates=None)
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            variables_averages_op = variable_averages.apply(variables_to_average)
            batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            self.train_step = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

    def add_result_ops(self):
        with tf.variable_scope('results'):
            labels = tf.argmax(self.ground_truth, 1)
            self.predictions = tf.nn.softmax(self.logits[0], name='predictions')
            correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
            self.top3 = in_top_k(self.predictions, labels, 3)
            self.top5 = in_top_k(self.predictions, labels, 5)

    def add_meta_nn(self):
        with tf.variable_scope('meta_NN'):
            lat_input = tf.reshape(self.lat_placeholder, (-1, 1)) / 90
            lng_input = tf.reshape(self.lng_placeholder, (-1, 1)) / 180
            week_input = tf.reshape(self.week_placeholder, (-1, 1)) / 25 - 1
            net = tf.concat(1, [lat_input, lng_input, week_input])
            # 3
            net = slim.ops.fc(net, 50, restore=False)
            net = slim.ops.fc(net, 50, restore=False)
        return net

    def build_graph(self, for_training=True):
        print("Building graph...")
        self.add_image_pre_processing()
        self.add_image_distortion()
        extra = self.add_meta_nn()
        self.logits = inception.inference(self.inception_input, self.class_count, extra_to_last_layer=extra,
                                          for_training=for_training, restore_logits=not for_training)
        self.add_train_step()
        self.add_result_ops()

    def init_fresh_model(self, sess):
        print("Loading original inception...")
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(ORIGINAL_INCEPTION_CKPT_DIR).model_checkpoint_path
        variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt)

    def load_last_checkpoint(self, sess, checkpoint=None, saver=None):
        saver = saver if saver else tf.train.Saver()
        last_checkpoint = checkpoint if checkpoint else tf.train.latest_checkpoint(self.save_path)
        if last_checkpoint:
            print("Restoring checkpoint...")
            saver.restore(sess, last_checkpoint)
        else:
            self.init_fresh_model(sess)

    def get_feed_dict(self, points):
        images, metas, labels, _ = points
        return {
            self.inception_input: images,
            self.ground_truth: labels,
            self.lat_placeholder: [const_for_none(meta['lat']) for meta in metas],
            self.lng_placeholder: [const_for_none(meta['lng']) for meta in metas],
            self.week_placeholder: [const_for_none(meta['week']) for meta in metas],
        }

    def evaluate(self, sess, dataset):
        with tf.variable_scope('evaluation'):
            hits, hits3, hits5 = 0, 0, 0
            i = 0
            for points in dataset.iter_per_part(100):
                samples = len(points[0])
                i += samples
                loss, acc, top3, top5 = sess.run([self.total_loss, self.accuracy, self.top3, self.top5], feed_dict=self.get_feed_dict(points))
                assert not np.isnan(loss), 'Model diverged with loss = NaN'
                hits += int(acc * samples)
                hits3 += int(top3 * samples)
                hits5 += int(top5 * samples)
                print("\r>>> Evaluation: {} / {} ({:.2f}%)".format(i, dataset.size, hits / i * 100), end="")

            accuracy = hits / dataset.size

            summary = tf.Summary()
            summary.value.add(tag='Accuracy', simple_value=accuracy)
            summary.value.add(tag='Recall @ 3', simple_value=hits3 / dataset.size)
            summary.value.add(tag='Recall @ 5', simple_value=hits5 / dataset.size)
        return accuracy, summary

    def train(self, sess, batch_size=20, evaluate_every=500, save_every=2000, checkpoint=None):
        summary_writer = tf.train.SummaryWriter(self.tensor_board_path, sess.graph, flush_secs=30)
        saver = tf.train.Saver(max_to_keep=2)
        self.load_last_checkpoint(sess, saver=saver, checkpoint=checkpoint)
        print("Training...")
        while True:
            sess.run(self.global_step.assign_add(1))
            step = int(self.global_step.value().eval())
            print("\r>>> Step: {}".format(int(step)), end="")

            points = self._data_set.train.get_batch(batch_size)
            self.train_step.run(feed_dict=self.get_feed_dict(points))

            if step % evaluate_every == 1:
                accuracy, summary = self.evaluate(sess, self._data_set.validation)
                print("\r>>> Step: {}, epoch: {:.2f}, LR: {:.6f}, accuracy: {:.2f}%".format(
                    step, step * batch_size / self._data_set.train.size,
                    self.learning_rate.eval(), accuracy * 100))
                summary_writer.add_summary(summary, step)

            if step % save_every == 0:
                path = os.path.join(self.save_path, "checkpoint")
                ensure_dir_exists(self.save_path)
                saver.save(sess, path, global_step=step)


def export():
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            model = InceptionModel(ds)
            model.build_graph(for_training=False)
            model.load_last_checkpoint(sess)

            output_file = os.path.join(MODEL_DIR, str(model) + ".pb")
            graph_def = graph.as_graph_def()
            graph_def = convert_variables_to_constants(sess, graph_def, ["results/predictions"])
            with gfile.FastGFile(output_file, 'wb') as f:
                f.write(graph_def.SerializeToString())


# ds = FlowerCheckerDataSet()
# download_flowerchecker_dataset("datasets/flowerchecker/real_dataset.json")
ds = FlowerCheckerDataSet(file_name='dataset_v2_small.json')
ds.prepare_data(validation_size=0.05)

export()


if False:
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = InceptionModel(ds)
            model.build_graph()
            ds.validation.pre_process_image = lambda img: model.pre_process_image(sess, img)
            ds.train.pre_process_image = lambda img: model.distort_image(sess, img)
            model.train(sess)
