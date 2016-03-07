from image_recognition.dataset import FlowerCheckerDataSet
from image_recognition.utils import *


class Recognizer:
    def __init__(self, data_set, name):
        self.data_set = data_set
        self.tensor_board_path = os.path.join("/tmp/plant_recognition", name)

        model_dir_name = maybe_download_inception_and_extract()
        self.graph = load_google_inception_graph(model_dir_name)
        self.train_step, self.cross_entropy_mean, self.accuracy, self.keep = add_final_training_ops(
            self.graph, data_set.class_count,
            optimizer=tf.train.AdamOptimizer, learning_rate=1e-4)

        self.image_data_placeholder = self.graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)
        self.bottleneck_tensor = self.graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
        self.ground_truth_placeholder = self.graph.get_tensor_by_name(GROUND_TRUTH_TENSOR_NAME)

    def get_bottlenecks(self, sess, data):
        data, labels, identificators = data
        return compute_bottlenecks(sess, self.graph, data, identificators,
                                   self.image_data_placeholder, self.bottleneck_tensor)

    def evaluate(self, sess, data, summaries=None, print_string=None):
        test_bottlenecks = self.get_bottlenecks(sess, data)

        ops = [self.accuracy]
        if summaries is not None:
            ops.append(summaries)
        results = sess.run(ops, feed_dict={
            self.bottleneck_tensor: test_bottlenecks,
            self.ground_truth_placeholder: data[1],
            self.keep: 1,
        })
        if print_string is not None:
            print("{} accuracy {:.3f}%".format(print_string, results[0] * 100))
        if summaries is not None:
            return results
        return results[0]

    def train(self, iterations=1000000, batch_size=100, evaluate_every=100):
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter(self.tensor_board_path, sess.graph_def)
            tf.scalar_summary("accuracy", self.accuracy)
            summaries = tf.merge_all_summaries()

            sess.run(tf.initialize_all_variables())
            for i in range(iterations):
                batch = self.data_set.train.get_batch(batch_size)
                bottlenecks = self.get_bottlenecks(sess, batch)
                self.train_step.run(feed_dict={
                    self.bottleneck_tensor: bottlenecks,
                    self.ground_truth_placeholder: batch[1],
                    self.keep: 0.5,
                })

                if i % evaluate_every == 0:
                    accuracy, summary_str = self.evaluate(sess, self.data_set.validation.get_all(), summaries=summaries)
                    writer.add_summary(summary_str, i * batch_size)
                    print("\r>>> Step: {}/{}, validation accuracy: {:.2f}; epochs: {}".format(
                        i, iterations, accuracy * 100, self.data_set.train.finished_epochs), end="")


FC_data_set = FlowerCheckerDataSet()
FC_data_set.prepare_data()

rec = Recognizer(FC_data_set, "final first try + dropout")
rec.train()
