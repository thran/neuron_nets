import os

import numpy as np
import re
import tensorflow as tf
import inception.inception_model as inception
from image_recognition.utils import in_top_k, ensure_dir_exists
from inception import slim
from image_recognition.fc_datasets import FlowerCheckerDataSet, prepare_inception_dirs


TENSOR_BOARD_DIR = "../../tenzor_board"
ORIGINAL_INCEPTION_CKPT_DIR = "models/inception-v3"
CACHE_DIR = "/home/thran/projects/cache"
INCEPTION_INPUT_SIZE = 299, 299
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


class InceptionModel:
    VERSION = 0.1

    def __init__(self, dataset, learning_rate=0.001):
        self._data_set = dataset
        self.class_count = dataset.class_count
        self.jpeg = tf.placeholder(dtype='string', name="jpeg")
        self.ground_truth = tf.placeholder(tf.float32, [None, self.class_count])

        self.processed_jpeq = None
        self.inception_input = None
        self.predictions = None
        self.cross_entropy = None
        self.train_step = None
        self.accuracy = None

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate, RMSPROP_DECAY,
                                                   momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)
        self.tensor_board_path = os.path.join(TENSOR_BOARD_DIR, str(self))
        self.save_path = os.path.join(CACHE_DIR, "checkpoints", str(self))

    def __str__(self):
        return "IncMod v{} - {} plants".format(self.VERSION, self._data_set.class_count)

    def add_image_pre_processing(self):
        image = tf.image.decode_jpeg(self.jpeg, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.sub(image, 0.5)
        image = tf.mul(image, 2.0)
        self.processed_jpeq = tf.image.resize_images(image, *INCEPTION_INPUT_SIZE)
        self.inception_input = tf.placeholder_with_default(
            tf.expand_dims(self.processed_jpeq, 0), shape=[None, INCEPTION_INPUT_SIZE[0], INCEPTION_INPUT_SIZE[1], 3])
        f = lambda img: model.pre_process_image(sess, img)
        self._data_set.train.pre_process_image = f
        self._data_set.validation.pre_process_image = f

    def pre_process_image(self, sess, image):
        image = tf.gfile.FastGFile(image, 'rb').read()
        return sess.run(self.processed_jpeq, feed_dict={self.jpeg: image})

    def add_train_step(self):
        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        #  TODO
        slim.losses.cross_entropy_loss(self.logits[0], self.ground_truth, label_smoothing=0.1, weight=1.0)
        slim.losses.cross_entropy_loss(self.logits[1], self.ground_truth, label_smoothing=0.1, weight=0.4, scope='aux_loss')

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)

        # Calculate the total loss for the current tower.
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)

        apply_gradient_op = self.optimizer.minimize(total_loss)

        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, num_updates=None)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)
        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        # Group all updates to into a single train op.
        self.train_step = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

    def add_result_ops(self):
        labels = tf.argmax(self.ground_truth, 1)
        self.predictions = tf.nn.softmax(self.logits[0], name='predictions')
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

    def build_graph(self):
        print("Building graph...")
        self.add_image_pre_processing()
        self.logits = inception.inference(self.inception_input, self.class_count, for_training=True, restore_logits=False)
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
            return int(last_checkpoint.split("-")[-1])
        self.init_fresh_model(sess)
        return 0

    def evaluate(self, sess, dataset):
        hits, hits3, hits5 = 0, 0, 0
        i = 0
        for points in dataset.iter_per_part(100):
            images, metas, labels, _ = points
            samples = len(images)
            i += samples
            print("\r>>> Evaluation: {} / {} ({:.2f}%)".format(i, dataset.size, hits / i * 100), end="")
            acc, top3, top5 = sess.run([self.accuracy, in_top_k(self.predictions, labels, 3),
                                        in_top_k(self.predictions, labels, 5)], feed_dict={
                self.inception_input: images,
                self.ground_truth: labels,
            })
            hits += acc * samples
            hits3 += top3 * samples
            hits5 += top5 * samples

        accuracy = hits / dataset.size

        summary = tf.Summary()
        summary.value.add(tag='Accuracy', simple_value=accuracy)
        summary.value.add(tag='Recall @ 3', simple_value=hits3 / dataset.size)
        summary.value.add(tag='Recall @ 5', simple_value=hits5 / dataset.size)
        return accuracy, summary

    def train(self, sess, batch_size=20, evaluate_every=500, save_every=2000, checkpoint=None):
        summary_writer = tf.train.SummaryWriter(self.tensor_board_path, sess.graph, flush_secs=30)
        saver = tf.train.Saver(max_to_keep=2)
        step = self.load_last_checkpoint(sess, saver=saver, checkpoint=checkpoint)
        print("Training...")
        while True:
            step += 1
            print("\r>>> Step: {}".format(step), end="")

            images, metas, labels, _ = self._data_set.train.get_batch(batch_size)
            self.train_step.run(feed_dict={
                self.inception_input: images,
                self.ground_truth: labels,
            })

            if step % evaluate_every == 1:
                accuracy, summary = self.evaluate(sess, self._data_set.validation)
                print("\n>>> Step: {}, epoch: {}, accuracy: {:.2f}%".format(
                     step, self._data_set.train.finished_epochs, accuracy * 100))
                summary_writer.add_summary(summary, step)

            if step % save_every == 0:
                path = os.path.join(self.save_path, "checkpoint")
                ensure_dir_exists(self.save_path)
                saver.save(sess, path, global_step=step)

ds = FlowerCheckerDataSet()
ds.prepare_data(validation_size=0.05)

if True:
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = InceptionModel(ds)
            model.build_graph()
            model.train(sess)
