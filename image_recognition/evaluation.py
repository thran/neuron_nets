import os
import tensorflow as tf
import numpy as np
from image_recognition.dataset import FlowerCheckerDataSet
import matplotlib.pyplot as plt

MODEL_DIR = "models"
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


class Model:
    def __init__(self, model_name):
        with tf.Session() as sess:
            model_filename = os.path.join(MODEL_DIR, model_name)
            with tf.gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
            self.graph = sess.graph
            self.result_tensor = self.graph.get_tensor_by_name('final_result:0')
            self.raw_predictions_tensor = self.result_tensor.op.inputs[0]
            self.image_data_placeholder = self.graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)

    def evaluate(self, data_set):
        with tf.Session() as sess:
            hits = 0
            for i, (image, label, _) in enumerate(data_set):
                print('\r>> Evaluating {} from {} {:.1f}%'.format(i + 1, data_set.size, (i + 1) / data_set.size * 100), end="")
                result = self.predict(sess, image)
                hits += 1 if np.argmax(result) == np.argmax(label) else 0
            print()
            print("Accuracy: {:.2f}".format(hits / data_set.size * 100))

    def predict(self, sess, image, with_raw_predictions=False):
        if with_raw_predictions:
            return sess.run([self.result_tensor, self.raw_predictions_tensor],
                            feed_dict={self.image_data_placeholder: image, "Placeholder:0": 1})
        return sess.run(self.result_tensor, feed_dict={self.image_data_placeholder: image, "Placeholder:0": 1})

    def show_random_image(self, sess, data_set):
        np.random.seed()
        image, label, id = data_set.get_random()
        prediction, raw_prediction = self.predict(sess, image, True)
        self.plot_image(image, prediction, data_set, raw_prediction, label)

    def plot_image(self, image, prediction, data_set, raw_prediction, true_label=None):
        plt.subplot(3, 1, 1)
        plt.imshow(tf.image.decode_jpeg(image).eval())
        if true_label is not None:
            plt.title("{}".format(data_set.get_class(true_label)))

        plt.subplot(3, 1, 2)
        plt.bar(range(data_set.class_count), prediction[0])
        plt.title("{} - {:.1f}%".format(data_set.get_class(prediction), 100 + max(raw_prediction[0]) / 2))
        plt.xticks(np.arange(data_set.class_count) + 0.5,
                   [data_set.get_class(i) for i in range(data_set.class_count)], rotation=90)

        plt.subplot(3, 1, 3)
        plt.bar(range(data_set.class_count), raw_prediction[0])

        plt.show()

    def show_random_images(self, sess, data_set):
        while True:
            self.show_random_image(sess, data_set)
            plt.close()

    def identify_plant(self, sess, jpeq_file_name, data_set):
        image = tf.gfile.FastGFile(jpeq_file_name, 'rb').read()
        prediction = self.predict(sess, image)
        self.plot_image(image, prediction, data_set)

FC_data_set = FlowerCheckerDataSet()
FC_data_set.prepare_data()
# FC_data_set.export_classes("classes.json")

model = Model("Hidden layers, hidden_neuron_counts:[2048], distort:{'brightness': 0.3, 'crop': 0.5, 'epochs': 10, 'flip': True}, cut_early:True.pb")
# model.evaluate(FC_data_set.test)
with tf.Session() as sess:
    model.show_random_images(sess, FC_data_set.test)
    # model.identify_plant(sess, "/home/thran/black_kytka.jpg", FC_data_set)
