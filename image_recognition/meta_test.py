
import tensorflow as tf
from image_recognition.utils import in_top_k, ensure_dir_exists, const_for_none
from image_recognition.fc_datasets import FlowerCheckerDataSet, prepare_inception_dirs
from inception.slim import slim


class MetaTEstModel:

    def __init__(self, dataset):
        self._data_set = dataset
        self.class_count = dataset.class_count
        self.lat_placeholder = tf.placeholder_with_default(tf.zeros([1], dtype=tf.float32), [None], name='lat_placeholder')
        self.lng_placeholder = tf.placeholder_with_default(tf.zeros([1], dtype=tf.float32), [None], name='lng_placeholder')
        self.week_placeholder = tf.placeholder_with_default(tf.zeros([1], dtype=tf.float32), [None], name='week_placeholder')
        self.ground_truth = tf.placeholder(tf.float32, [None, self.class_count])

    def build_graph(self):
        lat_input = tf.reshape(self.lat_placeholder, (-1, 1)) / 90
        lng_input = tf.reshape(self.lng_placeholder, (-1, 1)) / 180
        week_input = tf.reshape(self.week_placeholder, (-1, 1)) / 25 - 1
        net = tf.concat(1, [lat_input, lng_input, week_input])
        net = slim.ops.fc(net, 1050)
        net = slim.ops.fc(net, 1050)
        logits = slim.ops.fc(net, self.class_count)

        labels = tf.argmax(self.ground_truth, 1)
        self.predictions = tf.nn.softmax(logits, name='predictions')
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')

        cross_entropy = -tf.reduce_sum(self.ground_truth * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))
        cross_entropy = tf.reduce_mean(cross_entropy)
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    def get_feed_dict(self, points):
        images, metas, labels, _ = points
        return {
            self.ground_truth: labels,
            self.lat_placeholder: [const_for_none(meta['lat']) for meta in metas],
            self.lng_placeholder: [const_for_none(meta['lng']) for meta in metas],
            self.week_placeholder: [const_for_none(meta['week']) for meta in metas],
        }

    def evaluate(self, sess, dataset):
        accuracy = sess.run(self.accuracy, feed_dict=self.get_feed_dict(dataset.get_all()))
        return accuracy

    def train(self, sess, batch_size=200, evaluate_every=500, save_every=2000, checkpoint=None):
        sess.run(tf.initialize_all_variables())
        step = 0
        while True:
            step +=1
            print("\r>>> Step: {}".format(int(step)), end="")

            points = self._data_set.train.get_batch(batch_size)
            self.train_step.run(feed_dict=self.get_feed_dict(points))

            if step % evaluate_every == 1:
                accuracy = self.evaluate(sess, self._data_set.validation)
                print("\r>>> Step: {}, epoch: {:.2f}, accuracy: {:.2f}%".format(
                    step, step * batch_size / self._data_set.train.size, accuracy * 100))

ds = FlowerCheckerDataSet()
# ds = FlowerCheckerDataSet(file_name='dataset_v2_small.json')
ds.prepare_data(validation_size=0.05)

if True:
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = MetaTEstModel(ds)
            model.build_graph()
            model.train(sess)
