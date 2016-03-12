import inspect

from tensorflow.python.platform import gfile

from image_recognition.dataset import FlowerCheckerDataSet
from image_recognition.utils import *

TENSOR_BOARD_DIR = "../tenzor_board"
CACHE_DIR = "/home/thran/projects/cache"
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
INPUT_HEIGHT, INPUT_WIDTH = 299, 299
MODEL_DIR = "models"

EARLY_CUT = 'mixed_9/join:0'
EARLY_CUTS = {
    1: 'mixed_9/join:0',
    2: 'mixed_8/join:0',
}


class NetEnd:
    name = "Abstract Net end"

    def __init__(self, learning_rate=1e-4, optimizer=None, distort=None, cut_early=False):
        self.graph = None
        self.class_count = None
        self.image_data_placeholder = None
        self.resized_image_data_placeholder = None
        self.distorted_image_tensor = None
        self.bottleneck_tensor_size = BOTTLENECK_TENSOR_SIZE
        self.bottleneck_tensor = None
        self.predictions = None
        self.keep_prob_placeholder = None
        self.ground_truth_placeholder = None
        self.cross_entropy = None
        self.accuracy = None
        self._learning_rate = learning_rate
        self._optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer
        self._distort = distort
        self._cut_early = cut_early
        self.cache_dir = CACHE_DIR

        self.train_step = None

    def prepare(self, graph, class_count):
        self.graph = graph
        self.prepare_tensors()
        if self._cut_early:
            self.cut_early()
        self.class_count = class_count
        if self._distort is not None:
            self.distorted_image_tensor = add_input_distortions(
                self.image_data_placeholder, INPUT_HEIGHT, INPUT_WIDTH, 3, min_crop=self._distort["crop"],
                random_brightness=self._distort["brightness"], flip_left_right=self._distort["flip"])
        self.add_end()
        self.add_train_step()

    def add_end(self):
        pass

    def distort_images(self, sess, images):
        distorted = []
        for image in images:
            distorted.append(sess.run(self.distorted_image_tensor, feed_dict={self.image_data_placeholder: image}))
        return distorted

    def add_train_step(self):
        self.cross_entropy = -tf.reduce_sum(self.ground_truth_placeholder *
                                            tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))
        cross_entropy_mean = tf.reduce_mean(self.cross_entropy)
        self.train_step = self._optimizer(self._learning_rate).minimize(cross_entropy_mean)

        labels = tf.argmax(self.ground_truth_placeholder, 1)
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.scalar_summary("accuracy", self.accuracy)
        # tf.histogram_summary("probabilities", self.predictions)
        tf.scalar_summary("top3", in_top_k(self.predictions, labels, 3))
        tf.scalar_summary("top5", in_top_k(self.predictions, labels, 5))
        tf.scalar_summary("top10", in_top_k(self.predictions, labels, 10))

    def __str__(self):
        if self.name == "Abstract Net end":
            raise AttributeError("Model name not specified")
        s = self.name
        (args, _, _, defaults) = inspect.getargspec(self.__init__)
        if not defaults:
            args, defaults = tuple(), tuple()
        args += ("distort", "cut_early")
        defaults += (None, False)
        s += "".join([", {}:{}".format(a, represent(getattr(self, "_" + a))) for a, d in zip(args[-len(defaults):], defaults)
                      if getattr(self, "_" + a) != d])
        return s

    def __repr__(self):
        return hash_str(str(self))

    def prepare_tensors(self):
        self.bottleneck_tensor = self.graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
        self.image_data_placeholder = self.graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)
        self.resized_image_data_placeholder = self.graph.get_tensor_by_name(RESIZED_INPUT_TENSOR_NAME)
        self.keep_prob_placeholder = tf.placeholder("float")
        self.ground_truth_placeholder = tf.placeholder(tf.float32, [None, self.class_count])

    def cut_early(self):
        cut_name = EARLY_CUTS[self._cut_early] if type(self._cut_early) == int else EARLY_CUT
        self.bottleneck_tensor = self.graph.get_tensor_by_name(cut_name)
        self.bottleneck_tensor_size = 1
        for s in self.bottleneck_tensor.get_shape():
            self.bottleneck_tensor_size *= int(s)

        if self.bottleneck_tensor_size > 80000:
            self.bottleneck_tensor = tf.nn.max_pool(self.bottleneck_tensor,
                                                    ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.bottleneck_tensor_size //= 4
        self.bottleneck_tensor = tf.reshape(self.bottleneck_tensor, (1, self.bottleneck_tensor_size))

        self.cache_dir = os.path.join(CACHE_DIR, cut_name.replace('/', "#"))

    def get_bottlenecks(self, sess, data, evaluation=False, epoch=0):
        data, labels, identificators = data
        if self._distort and not evaluation:
            cache_dir = os.path.join(self.cache_dir, represent(self._distort), str(epoch // self._distort["epochs"]))
            return compute_bottlenecks(sess, data, identificators, self.resized_image_data_placeholder,
                                       self.bottleneck_tensor, cache_dir, prepare_function=self.distort_images)

        return compute_bottlenecks(sess, data, identificators,
                                   self.image_data_placeholder, self.bottleneck_tensor, self.cache_dir)


class SimpleNetEnd(NetEnd):
    name = "Simple"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_end(self):
        layer_weights = tf.Variable(tf.truncated_normal([self.bottleneck_tensor_size, self.class_count], stddev=0.001))
        layer_biases = tf.Variable(tf.zeros([self.class_count]))
        dropout = tf.nn.dropout(self.bottleneck_tensor, self.keep_prob_placeholder)
        logits = tf.matmul(dropout, layer_weights) + layer_biases
        self.predictions = tf.nn.softmax(logits, name="final_result")


class HiddenLayersNetEnd(NetEnd):
    name = "Hidden layers"

    def __init__(self, hidden_neuron_counts=None, **kwargs):
        super().__init__(**kwargs)
        self._hidden_neuron_counts = hidden_neuron_counts

    def add_end(self):
        last_count = self.bottleneck_tensor_size
        layer = self.bottleneck_tensor

        for count in self._hidden_neuron_counts:
            layer_weights = tf.Variable(tf.truncated_normal([last_count, count], stddev=0.001))
            layer_biases = tf.Variable(tf.zeros([count]))
            layer = tf.nn.relu(tf.matmul(layer, layer_weights) + layer_biases)
            last_count = count

        layer_weights = tf.Variable(tf.truncated_normal([last_count, self.class_count], stddev=0.001))
        layer_biases = tf.Variable(tf.zeros([self.class_count]))

        dropout = tf.nn.dropout(layer, self.keep_prob_placeholder)
        logits = tf.matmul(dropout, layer_weights) + layer_biases
        self.predictions = tf.nn.softmax(logits, name="final_result")


class Trainer:
    def __init__(self, data_set, net_end):
        self.data_set = data_set
        model_dir_name = maybe_download_inception_and_extract(MODEL_DIR)

        self.graph = load_google_inception_graph(model_dir_name)
        self.ne = net_end
        self.ne.prepare(self.graph, self.data_set.class_count)
        self.tensor_board_path = os.path.join(TENSOR_BOARD_DIR, str(self.ne))
        self.save_path = os.path.join(CACHE_DIR, "checkpoints", repr(self.ne))

    def evaluate(self, sess, data, summaries=None, print_string=None):
        test_bottlenecks = self.ne.get_bottlenecks(sess, data, evaluation=True)

        ops = [self.ne.accuracy]
        if summaries is not None:
            ops.append(summaries)
        results = sess.run(ops, feed_dict={
            self.ne.bottleneck_tensor: test_bottlenecks,
            self.ne.ground_truth_placeholder: data[1],
            self.ne.keep_prob_placeholder: 1,
        })
        if print_string is not None:
            print("{} accuracy {:.3f}%".format(print_string, results[0] * 100))
        if summaries is not None:
            return results
        return results[0]

    def train(self, batch_size=50, evaluate_every=200, save_every=5000, checkpoint=None):
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter(self.tensor_board_path, sess.graph_def, flush_secs=30)
            summaries = tf.merge_all_summaries()
            saver = tf.train.Saver()
            i = self.load_last_checkpoint(sess, checkpoint, saver)
            if i == 0:
                sess.run(tf.initialize_all_variables())
            while True:
                i += 1
                batch = self.data_set.train.get_batch(batch_size)
                bottlenecks = self.ne.get_bottlenecks(sess, batch, epoch=self.data_set.train.finished_epochs)
                self.ne.train_step.run(feed_dict={
                    self.ne.bottleneck_tensor: bottlenecks,
                    self.ne.ground_truth_placeholder: batch[1],
                    self.ne.keep_prob_placeholder: 0.5,
                })

                if i % evaluate_every == 0:
                    accuracy, summary_str = self.evaluate(sess, self.data_set.validation.get_all(), summaries=summaries)
                    writer.add_summary(summary_str, i * batch_size)
                    print("\r>>> Step: {}, validation accuracy: {:.2f}; epochs: {}".format(
                        i, accuracy * 100, self.data_set.train.finished_epochs), end="")

                if (i + 1) % save_every == 0:
                    path = os.path.join(self.save_path, "checkpoint")
                    ensure_dir_exists(self.save_path)
                    saver.save(sess, path, global_step=i + 1)

    def load_last_checkpoint(self, sess, checkpoint=None, saver=None):
        saver = saver if saver else tf.train.Saver()
        last_checkpoint = checkpoint if checkpoint else tf.train.latest_checkpoint(self.save_path)
        if last_checkpoint:
            saver.restore(sess, last_checkpoint)
            i = int(last_checkpoint.split("-")[-1])
        else:
            i = 0
        return i

    def export(self):
        output_file = os.path.join(MODEL_DIR, str(self.ne) + ".pb")
        print(output_file)
        with gfile.FastGFile(output_file, 'wb') as f:
            f.write(self.graph.as_graph_def().SerializeToString())


FC_data_set = FlowerCheckerDataSet()
FC_data_set.prepare_data()

if False:
    ne = SimpleNetEnd()
    # ne = HiddenLayersNetEnd([2048], distort={"crop": 0.5, "brightness": 0.3, "flip": True, "epochs": 10})
    trainer = Trainer(FC_data_set, ne)
    trainer.train(evaluate_every=50, save_every=100)

if True:
    # ne = SimpleNetEnd(cut_early=True)
    ne = HiddenLayersNetEnd([2048], learning_rate=1e-4,
                            distort={"crop": 0.5, "brightness": 0.3, "flip": True, "epochs": 10}, cut_early=True)
    trainer = Trainer(FC_data_set, ne)
    print(ne, repr(ne))
    # trainer.train()
    with tf.Session() as sess:
        trainer.load_last_checkpoint(sess)
        trainer.export()

if False:
    ne = HiddenLayersNetEnd([2048], learning_rate=1e-5, cut_early=2)
    trainer = Trainer(FC_data_set, ne)
    print(ne, repr(ne))
    trainer.train()
