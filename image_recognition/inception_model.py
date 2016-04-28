import numpy as np
import re
import tensorflow as tf
import inception.inception_model as inception
from image_recognition.utils import in_top_k
from inception import slim
from image_recognition.fc_datasets import FlowerCheckerDataSet

ORIGINAL_INCEPTION_CKPT_DIR = "models/inception-v3"
INCEPTION_INPUT_SIZE = 299, 299
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


class InceptionModel:

    def __init__(self, dataset, learning_rate=0.1):
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

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate, RMSPROP_DECAY, momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)

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
        return sess.run(self.processed_jpeq, feed_dict={self.jpeg: image})

    def add_inception_end(self):
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

    def add_train_step(self):
        cross_entropy = -tf.reduce_sum(self.ground_truth *
                                       tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))
        self.cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
        self.train_step = self.optimizer.minimize(self.cross_entropy)

    def add_result_ops(self):
        labels = tf.argmax(self.ground_truth, 1)
        self.predictions = tf.nn.softmax(self.logits[0], name='predictions')
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

        tf.scalar_summary("accuracy", self.accuracy)
        tf.scalar_summary("top3", in_top_k(self.predictions, labels, 3))
        tf.scalar_summary("top5", in_top_k(self.predictions, labels, 5))
        tf.scalar_summary("top10", in_top_k(self.predictions, labels, 10))

    def build_graph(self):
        print("Building graph...")
        self.add_image_pre_processing()
        self.logits = inception.inference(self.inception_input, self.class_count, for_training=True, restore_logits=False)
        # self.add_train_step()
        self.add_inception_end()
        self.add_result_ops()

    def init_fresh_model(self, sess):
        print("Loading graph...")
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(ORIGINAL_INCEPTION_CKPT_DIR).model_checkpoint_path
        variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt)

    def evaluate(self, sess, dataset):
        hits = 0
        i = 0
        for points in dataset.iter_per_part(100):
            images, metas, labels, _ = points
            samples = len(images)
            i += samples
            print("\r>>> Evaluation: {} / {} ({:.2f}%)".format(i, dataset.size, hits / i * 100), end="")
            acc = sess.run(self.accuracy, feed_dict={
                self.inception_input: images,
                self.ground_truth: labels,
            })
            hits += acc * samples

        return hits / dataset.size

    def train(self, sess):
        print("Training...")
        i = 0
        while True:
            i += 1

            if i % 200 == 0:
                accuracy = self.evaluate(sess, self._data_set.validation)
                print("\n>>> Step: {}, epoch: {}, accuracy: {:.2f}%".format(
                     i, self._data_set.train.finished_epochs, accuracy * 100))

            print("\r>>> Step: {}".format(i), end="")

            images, metas, labels, _ = self._data_set.train.get_batch(20)
            self.train_step.run(feed_dict={
                self.inception_input: images,
                self.ground_truth: labels,
            })

import matplotlib.pyplot as plt

ds = FlowerCheckerDataSet()
ds.prepare_data(test_size=0, validation_size=0.03)

if True:
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = InceptionModel(ds)
            model.build_graph()
            model.init_fresh_model(sess)
            model.train(sess)

plt.show()
