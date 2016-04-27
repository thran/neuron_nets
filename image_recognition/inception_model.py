import numpy as np
import tensorflow as tf
import inception.inception_model as inception
from image_recognition.utils import in_top_k
from inception import slim
from image_recognition.fc_datasets import FlowerCheckerDataSet

ORIGINAL_INCEPTION_CKPT_DIR = "models/inception-v3"
INCEPTION_INPUT_SIZE = 299, 299


class InceptionModel:

    def __init__(self, dataset, learning_rate=2e-5, optimizer=None):
        self._data_set = dataset
        self._learning_rate = learning_rate
        self._optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer
        self.class_count = dataset.class_count
        self.jpeg = tf.placeholder(dtype='string', name="jpeg")
        self.ground_truth = tf.placeholder(tf.float32, [None, self.class_count])

        self.processed_jpeq = None
        self.inception_input = None
        self.predictions = None
        self.cross_entropy = None
        self.train_step = None
        self.accuracy = None

    def add_image_pre_processing(self):
        decoded_jpeg = tf.image.decode_jpeg(self.jpeg, channels=3)
        self.processed_jpeq = tf.image.resize_images(decoded_jpeg, *INCEPTION_INPUT_SIZE)
        self.inception_input = tf.placeholder_with_default(
            tf.expand_dims(self.processed_jpeq, 0), shape=[None, INCEPTION_INPUT_SIZE[0], INCEPTION_INPUT_SIZE[1], 3])
        f = lambda img: model.pre_process_image(sess, img)
        self._data_set.train.pre_process_image = f
        self._data_set.validation.pre_process_image = f

    def pre_process_image(self, sess, image):
        return sess.run(self.processed_jpeq, feed_dict={self.jpeg: image})

    def add_train_step(self):
        cross_entropy = -tf.reduce_sum(self.ground_truth *
                                       tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))
        self.cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
        self.train_step = self._optimizer(self._learning_rate).minimize(self.cross_entropy)

        labels = tf.argmax(self.ground_truth, 1)
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        tf.scalar_summary("accuracy", self.accuracy)
        tf.scalar_summary("top3", in_top_k(self.predictions, labels, 3))
        tf.scalar_summary("top5", in_top_k(self.predictions, labels, 5))
        tf.scalar_summary("top10", in_top_k(self.predictions, labels, 10))

    def build_graph(self):
        self.add_image_pre_processing()
        logits, _ = inception.inference(self.inception_input, self.class_count, for_training=True, restore_logits=False)
        self.predictions = tf.nn.softmax(logits, name='predictions')
        self.add_train_step()

    def init_fresh_model(self, sess):
        sess.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(ORIGINAL_INCEPTION_CKPT_DIR).model_checkpoint_path
        variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt)

    def evaluate(self, sess, dataset):
        hits = 0
        for i, (image, meta, label, _) in enumerate(dataset):
            print("\r>>> Evaluation: {} / {}".format(i, dataset.size), end="")
            hits += sess.run(self.accuracy, feed_dict={
                self.inception_input: [image],
                self.ground_truth: [label],
            })
        return hits / dataset.size

    def train(self, sess):
        i = 0
        while True:
            i += 1

            if i % 500 == 0:
                accuracy = self.evaluate(sess, self._data_set.validation)
                print("\n>>> Step: {}, epoch: {}, accuracy: {:.2f}".format(
                     i, self._data_set.finished_epochs, accuracy * 100))

            print("\r>>> Step: {}".format(i), end="")

            images, metas, labels, _ = self._data_set.train.get_batch(20)
            self.train_step.run(feed_dict={
                self.inception_input: images,
                self.ground_truth: labels,
            })


ds = FlowerCheckerDataSet()
ds.prepare_data(test_size=0, validation_size=0.03)

if True:
    with tf.Session() as sess:
        model = InceptionModel(ds)
        model.build_graph()
        model.init_fresh_model(sess)
        model.train(sess)


