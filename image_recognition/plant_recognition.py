import inspect
from image_recognition.dataset import FlowerCheckerDataSet
from image_recognition.utils import *

CACHE_DIR = "cache"
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
INPUT_HEIGHT, INPUT_WIDTH = 299, 299


class NetEnd:
    name = "Abstract Net end"

    def __init__(self, learning_rate=1e-4, optimizer=None, distort=None):
        self.graph = None
        self.class_count = None
        self.image_data_placeholder = None
        self.resized_image_data_placeholder = None
        self.distorted_image_tensor = None
        self.bottleneck_tensor = None
        self.predictions = None
        self.keep_prob_placeholder = None
        self.ground_truth_placeholder = None
        self.cross_entropy = None
        self.accuracy = None
        self._learning_rate = learning_rate
        self._optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer
        self._distort = distort

        self.train_step = None

    def prepare(self, graph, class_count):
        self.graph = graph
        self.prepare_tensors()
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
        self.cross_entropy = -tf.reduce_sum(self.ground_truth_placeholder * tf.log(self.predictions))
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
        args += ("learning_rate", "optimizer", "distort")
        defaults += (1e-4, tf.train.AdamOptimizer, None)
        s += "".join([", {}:{}".format(a, repr(getattr(self, "_" + a))) for a, d in zip(args[-len(defaults):], defaults)
                      if getattr(self, "_" + a) != d])
        return s

    def prepare_tensors(self):
        self.bottleneck_tensor = self.graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
        self.image_data_placeholder = self.graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)
        self.resized_image_data_placeholder = self.graph.get_tensor_by_name(RESIZED_INPUT_TENSOR_NAME)
        self.keep_prob_placeholder = tf.placeholder("float")
        self.ground_truth_placeholder = tf.placeholder(tf.float32, [None, self.class_count])

    def get_bottlenecks(self, sess, data, evaluation=False):
        data, labels, identificators = data
        if self._distort and not evaluation:
            data = self.distort_images(sess, data)
            return compute_bottlenecks(sess, data, identificators,
                                       self.resized_image_data_placeholder, self.bottleneck_tensor)

        cache_dir = CACHE_DIR
        return compute_bottlenecks(sess, data, identificators,
                                   self.image_data_placeholder, self.bottleneck_tensor, cache_dir)


class SimpleNetEnd(NetEnd):
    name = "Simple"

    def add_end(self):
        layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, self.class_count], stddev=0.001))
        layer_biases = tf.Variable(tf.zeros([self.class_count]))
        dropout = tf.nn.dropout(self.bottleneck_tensor, self.keep_prob_placeholder)
        logits = tf.matmul(dropout, layer_weights) + layer_biases
        self.predictions = tf.nn.softmax(logits)


class HiddenLayersNetEnd(NetEnd):
    name = "Hidden layers"

    def __init__(self, hidden_neuron_counts=None, **kwargs):
        super().__init__(**kwargs)
        self._hidden_neuron_counts = hidden_neuron_counts

    def add_end(self):
        last_count = BOTTLENECK_TENSOR_SIZE
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
        self.predictions = tf.nn.softmax(logits)


class Recognizer:
    def __init__(self, data_set, net_end):
        self.data_set = data_set
        model_dir_name = maybe_download_inception_and_extract()

        self.graph = load_google_inception_graph(model_dir_name)
        self.ne = net_end
        self.ne.prepare(self.graph, self.data_set.class_count)
        self.tensor_board_path = os.path.join("/tmp/plant_recognition", str(self.ne))

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

    def train(self, iterations=30000, batch_size=50, evaluate_every=200):
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter(self.tensor_board_path, sess.graph_def, flush_secs=30)
            summaries = tf.merge_all_summaries()

            sess.run(tf.initialize_all_variables())
            for i in range(iterations):
                batch = self.data_set.train.get_batch(batch_size)
                bottlenecks = self.ne.get_bottlenecks(sess, batch)
                self.ne.train_step.run(feed_dict={
                    self.ne.bottleneck_tensor: bottlenecks,
                    self.ne.ground_truth_placeholder: batch[1],
                    self.ne.keep_prob_placeholder: 0.5,
                })

                if i % evaluate_every == 0:
                    accuracy, summary_str = self.evaluate(sess, self.data_set.validation.get_all(), summaries=summaries)
                    writer.add_summary(summary_str, i * batch_size)
                    print("\r>>> Step: {}/{}, validation accuracy: {:.2f}; epochs: {}".format(
                        i, iterations, accuracy * 100, self.data_set.train.finished_epochs), end="")


FC_data_set = FlowerCheckerDataSet()
FC_data_set.prepare_data()

if True:
    ne = HiddenLayersNetEnd([2048], distort={"crop": 0.5, "brightness": 0.3, "flip": True})
    rec = Recognizer(FC_data_set, ne)
    print(ne)
    rec.train(evaluate_every=20)
else:
    ne = HiddenLayersNetEnd([2048], learning_rate=0.001)
    rec = Recognizer(FC_data_set, ne)
    print(ne)
    rec.train()
