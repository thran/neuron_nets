import json
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from image_recognition.dataset import CertaintyDataSet
import tfdeploy as td


def reliability_curve(y_true, y_score, bins=10):
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos


class CertaintyNN:
    def __init__(self, input_size, output_size, hidden_layer_size=300):
        self.graph = None
        self.input_size = input_size
        self.row_predictions = tf.placeholder(tf.float32, [None, self.input_size], name="row_predictions")
        self.truth = tf.placeholder(tf.float32, [None, output_size], name="truth")
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope("normalization") as scope:
            self.normalized = self.row_predictions / 1

        with tf.name_scope("first_layer") as scope:
            W1 = tf.Variable(tf.truncated_normal([self.input_size, hidden_layer_size], stddev=0.001))
            b1 = tf.Variable(tf.zeros([hidden_layer_size]))
            layer = tf.nn.relu(tf.matmul(self.normalized, W1) + b1)

        # with tf.name_scope("second_layer") as scope:
        #     W2 = tf.Variable(tf.truncated_normal([hidden_layer_size, hidden_layer_size], stddev=0.001))
        #     b2 = tf.Variable(tf.zeros([hidden_layer_size]))
        #     layer = tf.nn.dropout(tf.nn.relu(tf.matmul(layer, W2) + b2), keep_prob=self.keep_prob)

        with tf.name_scope("output") as scope:
            W3 = tf.Variable(tf.truncated_normal([hidden_layer_size, output_size], stddev=0.001))
            b3 = tf.Variable(tf.zeros([output_size]))
            self.output = tf.sigmoid(tf.matmul(layer, W3) + b3, name="certainties")

        with tf.name_scope("RMSE") as scope:
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.truth - self.output)), name="rmse")

        with tf.name_scope("train") as scope:
            self.train_step = tf.train.AdamOptimizer(5e-5).minimize(self.rmse)

    def train(self, sess, data_set, steps=15000, batch_size=1000):
        sess.run(tf.initialize_all_variables())
        for i in range(steps):
            batch = data_set.train.get_batch(batch_size)
            self.train_step.run(feed_dict={
                self.row_predictions: batch[0],
                self.truth: batch[1],
                self.keep_prob: 0.5,
            })

            if i % 100 == 0:
                data = data_set.validation.get_all()
                rmse = sess.run(self.rmse, feed_dict={
                    self.row_predictions: data[0],
                    self.truth: data[1],
                    self.keep_prob: 1,
                })
                print("\r>>> Step: {}, RMSE: {:.5f}; epochs: {}".format(
                    i, rmse, data_set.train.finished_epochs), end="")

    def evaluate(self, sess, data_set):
        data = data_set.get_all()
        outputs = sess.run(self.output, feed_dict={
            self.row_predictions: data[0],
            self.keep_prob: 1,
        })

        for i in range(4):
            plt.figure()
            plt.subplot(311)
            for output, raws, truth in zip(outputs, data[0], data[1]):
                plt.plot(output[i], max(raws), ".", color="r" if truth[i] else "b")

            plt.subplot(312)
            x, y = reliability_curve([d[i] for d in data[1]], [d[i] for d in outputs])
            plt.plot(x, y)
            plt.plot([0, 1], [0, 1])

            plt.subplot(313)
            sns.distplot([d[i] for d in outputs])
            plt.xlim([0, 1])
        plt.show()

    def save(self, sess):
        model = td.Model()
        model.add(self.output, sess)
        model.save("models/certainty_model.pkl")


data_set = CertaintyDataSet()
data_set.prepare_data(test_size=0)

nn = CertaintyNN(input_size=len(data_set._data[0]), output_size=len(data_set._labels[0]))
with tf.Session() as sess:
    nn.train(sess, data_set)
    # nn.evaluate(sess, data_set.validation)
    nn.save(sess)
