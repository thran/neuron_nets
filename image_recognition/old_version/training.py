import inspect

from tensorflow.python.platform import gfile

from image_recognition.old_version.dataset import FlowerCheckerDataSet
from image_recognition.old_version.utils import *

TENSOR_BOARD_DIR = "../../tenzor_board"
CACHE_DIR = "/home/thran/projects/cache"
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
INPUT_HEIGHT, INPUT_WIDTH = 299, 299
MODEL_DIR = "../models"

CUT_TENSOR_NAME = 'mixed_9/join:0'


class NetEnd:
    name = "Abstract Net end"

    def __init__(self, learning_rate=1e-4, optimizer=None):
        self.graph = None
        self.class_count = None
        self.image_data_placeholder = None
        self.resized_image_data_placeholder = None
        self.distorted_image_tensor = None
        self.bottleneck_tensor_size = None
        self.bottleneck_tensor = None
        self.predictions = None
        self.keep_prob_placeholder = None
        self.ground_truth_placeholder = None
        self.cross_entropy = None
        self.accuracy = None
        self._learning_rate = learning_rate
        self._optimizer = optimizer if optimizer is not None else tf.train.AdamOptimizer
        self.cache_dir = CACHE_DIR
        self.has_meta = False

        self.train_step = None

    def prepare(self, graph, class_count):
        self.graph = graph
        self.prepare_tensors()
        self.cut()
        self.class_count = class_count
        self.add_end()
        self.add_train_step()
        self.distorted_image_tensor = add_input_distortions(self.image_data_placeholder, INPUT_HEIGHT, INPUT_WIDTH, 3,
                                                            min_crop=0.5, random_brightness=0.5, flip_left_right=True)

    def add_end(self):
        pass

    def add_train_step(self):
        self.cross_entropy = -tf.reduce_sum(self.ground_truth_placeholder *
                                            tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))
        cross_entropy_mean = tf.reduce_mean(self.cross_entropy)
        self.train_step = self._optimizer(self._learning_rate).minimize(cross_entropy_mean)

        labels = tf.argmax(self.ground_truth_placeholder, 1)
        correct_prediction = tf.equal(tf.argmax(self.predictions, 1), labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        tf.scalar_summary("accuracy", self.accuracy)
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
        s += "".join(
            [", {}:{}".format(a, represent(getattr(self, "_" + a))) for a, d in zip(args[-len(defaults):], defaults)
             if getattr(self, "_" + a) != d])
        return s

    def __repr__(self):
        return hash_str(str(self))

    def prepare_tensors(self):
        self.image_data_placeholder = self.graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)
        print(self.image_data_placeholder)
        self.resized_image_data_placeholder = self.graph.get_tensor_by_name(RESIZED_INPUT_TENSOR_NAME)
        self.keep_prob_placeholder = tf.placeholder("float")
        self.ground_truth_placeholder = tf.placeholder(tf.float32, [None, self.class_count])

    def cut(self):
        cut_tensor = self.graph.get_tensor_by_name(CUT_TENSOR_NAME)
        cut_tensor_size = 1
        for s in cut_tensor.get_shape():
            cut_tensor_size *= int(s)

        max_pool = tf.nn.max_pool(cut_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.bottleneck_tensor_size = cut_tensor_size // 4
        self.bottleneck_tensor = tf.reshape(max_pool, (1, self.bottleneck_tensor_size))

    def get_bottlenecks(self, sess, data, evaluation=False):
        data, labels, identificators = data
        if "distorted" in data[0][1]:
            return compute_bottlenecks(sess, [img for img, meta in data], identificators, self.resized_image_data_placeholder,
                                       self.bottleneck_tensor, self.cache_dir, prepare_function=self.distort_images)
        return compute_bottlenecks(sess, [img for img, meta in data], identificators,
                                   self.image_data_placeholder, self.bottleneck_tensor, self.cache_dir)

    def feed_meta(self, feed_dict, meta):
        return feed_dict

    def distort_images(self, sess, images):
        distorted = []
        for image in images:
            distorted.append(sess.run(self.distorted_image_tensor, feed_dict={self.image_data_placeholder: image}))
        return distorted


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


class HiddenLayersMetaNetEnd(NetEnd):
    name = "Hidden layers with meta 2.0"

    def __init__(self, hidden_neuron_counts=None, hidden_meta_counts=None, v="", **kwargs):
        super().__init__(**kwargs)
        self._hidden_neuron_counts = hidden_neuron_counts
        self._hidden_meta_counts = hidden_meta_counts
        self._v = v

        self.has_meta = True

    def add_end(self):
        self.lat_placeholder = tf.placeholder(tf.float32, [None], name='lat_placeholder')
        self.lng_placeholder = tf.placeholder(tf.float32, [None], name='lng_placeholder')
        self.week_placeholder = tf.placeholder(tf.float32, [None], name='week_placeholder')
        lat_input = tf.reshape(self.lat_placeholder, (-1, 1)) / 90
        lng_input = tf.reshape(self.lng_placeholder, (-1, 1)) / 180
        week_input = tf.reshape(self.week_placeholder, (-1, 1)) / 25 - 1
        layer_meta = tf.concat(1, [lat_input, lng_input, week_input])
        last_meta_count = 3
        for count in self._hidden_meta_counts:
            layer_weights = tf.Variable(tf.truncated_normal([last_meta_count, count], stddev=0.001))
            layer_biases = tf.Variable(tf.zeros([count]))
            self.layer_meta = layer_meta = tf.nn.relu(tf.matmul(layer_meta, layer_weights) + layer_biases)
            last_meta_count = count

        last_count = self.bottleneck_tensor_size
        layer = self.bottleneck_tensor
        for i, count in enumerate(self._hidden_neuron_counts):
            layer_weights = tf.Variable(tf.truncated_normal([last_count, count], stddev=0.001))
            layer_biases = tf.Variable(tf.zeros([count]))
            layer = tf.nn.relu(tf.matmul(layer, layer_weights) + layer_biases)
            last_count = count
            if i == 0:
                last_count = last_count + last_meta_count
                layer = tf.concat(1, [layer, layer_meta])

        layer_weights = tf.Variable(tf.truncated_normal([last_count, self.class_count], stddev=0.001))
        layer_biases = tf.Variable(tf.zeros([self.class_count]))

        dropout = tf.nn.dropout(layer, self.keep_prob_placeholder)
        logits = tf.matmul(dropout, layer_weights) + layer_biases
        self.predictions = tf.nn.softmax(logits, name="final_result")

    def feed_meta(self, feed_dict, data):
        feed_dict[self.lat_placeholder] = [const_for_none(meta['lat']) for img, meta in data]
        feed_dict[self.lng_placeholder] = [const_for_none(meta['lng']) for img, meta in data]
        feed_dict[self.week_placeholder] = [const_for_none(meta['week']) for img, meta in data]
        return feed_dict


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
        results = sess.run(ops, feed_dict=self.ne.feed_meta({
            self.ne.bottleneck_tensor: test_bottlenecks,
            self.ne.ground_truth_placeholder: data[1],
            self.ne.keep_prob_placeholder: 1,
        }, data[0]))
        if print_string is not None:
            print("{} accuracy {:.3f}%".format(print_string, results[0] * 100))
        if summaries is not None:
            return results
        return results[0]

    def compute_bottlenecks(self, data_set):
        with tf.Session() as sess:
            data = data_set.get_part(10000)
            while data is not None:
                bns = self.ne.get_bottlenecks(sess, data)
                del bns
                del data
                data = data_set.get_part(10000)
            del data

    def train(self, batch_size=50, evaluate_every=200, save_every=5000, checkpoint=None):
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter(self.tensor_board_path, sess.graph_def, flush_secs=30)
            summaries = tf.merge_all_summaries()

            saver = tf.train.Saver(max_to_keep=2)
            i = self.load_last_checkpoint(sess, checkpoint, saver)
            if i == 0:
                sess.run(tf.initialize_all_variables())

            while True:
                i += 1
                batch = self.data_set.train.get_batch(batch_size)
                bottlenecks = self.ne.get_bottlenecks(sess, batch)
                self.ne.train_step.run(feed_dict=self.ne.feed_meta({
                    self.ne.bottleneck_tensor: bottlenecks,
                    self.ne.ground_truth_placeholder: batch[1],
                    self.ne.keep_prob_placeholder: 0.5,
                }, batch[0]))

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

    def export(self, sess):
        output_file = os.path.join(MODEL_DIR, str(self.ne) + ".pb")
        print(output_file)
        graph_def = self.graph.as_graph_def()
        graph_def = convert_variables_to_constants(sess, graph_def, ["final_result"])
        # TODO - replace keep_placeholder
        with gfile.FastGFile(output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())

distortions = 0
# FC_data_set = FlowerCheckerDataSet(distortions=distortions)
FC_data_set = FlowerCheckerDataSet(file_name="dataset-big.json", dir_name="../datasets/flowerchecker", distortions=distortions)
FC_data_set.prepare_data(test_size=0, balanced_train=False)

if True:
    # ne = SimpleNetEnd(cut_early=True)
    # ne = HiddenLayersNetEnd([2048], learning_rate=1e-4)
    ne = HiddenLayersMetaNetEnd([2048], [50, 50], learning_rate=1e-4, v="big")
    trainer = Trainer(FC_data_set, ne)
    print(ne, repr(ne))
    if False:
        for i in range(0, distortions + 1):
            print(i)
            FC_data_set.train.finished_epochs = i
            trainer.compute_bottlenecks(FC_data_set.train)
            FC_data_set.train._position_part = 0
    # trainer.train()
    # with tf.Session() as sess:
    #     trainer.load_last_checkpoint(sess)
    #     trainer.export(sess)
