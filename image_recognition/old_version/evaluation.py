import json
from collections import defaultdict
import os
import tensorflow as tf
import tfdeploy as td
import numpy as np
from json import encoder

from image_recognition.fc_datasets import FlowerCheckerDataSet
import matplotlib.pyplot as plt
import seaborn as sns
encoder.FLOAT_REPR = lambda o: format(o, '.5f')

MODEL_DIR = "models"
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
LAT_TENSOR_NAME = 'lat_placeholder:0'
LNG_TENSOR_NAME = 'lng_placeholder:0'
WEEK_TENSOR_NAME = 'week_placeholder:0'


class CustomFlowerCheckerDataSet(FlowerCheckerDataSet):
    def _prepare_classes(self, data):
        fc = FlowerCheckerDataSet()
        fc.prepare_data()
        self._classes = fc._classes
        self.class_count = len(self._classes)

        new_data = defaultdict(lambda: [])
        for cls, plaints in data.items():
            cls = cls.split(" ")[0]
            cls = cls if cls in self._classes else ""
            new_data[cls] += plaints
        return new_data


def plot_image(point, prediction, data_set, raw_prediction, true_label=None, certainties=None):
    image, meta = point
    plt.subplot(1, 3, 1)
    plt.imshow(tf.image.decode_jpeg(image).eval())
    if true_label is not None:
        plt.title("{}".format(data_set.get_class(true_label)))

    plt.subplot(1, 3, 2)
    count = 10
    best = prediction[0].argsort()[-count:][::-1]
    plt.bar(range(count), prediction[0][best])
    plt.title("{} - {:.1f}%".format(data_set.get_class(prediction), 100 * certainties[0]))
    plt.xticks(np.arange(count) + 0.5,
               [data_set.get_class(int(i)) for i in best], rotation=90)

    plt.subplot(1, 3, 3)
    plt.title(certainties)
    sns.distplot(raw_prediction[0], rug=True)

    plt.show()


class CertaintyModel:
    def __init__(self):
        self.model = td.Model(os.path.join(MODEL_DIR, "certainty_model.pkl"))
        self.input, self.output = self.model.get("row_predictions", "output/certainties")

    def get_certainty(self, input):
        return self.output.eval({self.input: sorted(input[0])})


class Model:
    def __init__(self, model_name):
        self.certainty_model = CertaintyModel()
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
            self.lat_placeholder = self.graph.get_tensor_by_name(LAT_TENSOR_NAME)
            self.lng_placeholder = self.graph.get_tensor_by_name(LNG_TENSOR_NAME)
            self.week_placeholder = self.graph.get_tensor_by_name(WEEK_TENSOR_NAME)

    def evaluate(self, data_set):
        with tf.Session() as sess:
            hits = 0
            for i, (image, label, _) in enumerate(data_set):
                print('\r>> Evaluating {} from {} {:.1f}%'.format(i + 1, data_set.size, (i + 1) / data_set.size * 100), end="")
                result = self.predict(sess, image)
                if sum(label) > 0:
                    hits += 1 if np.argmax(result) == np.argmax(label) else 0
            print()
            print("Accuracy: {:.2f}".format(hits / data_set.size * 100))

    def predict(self, sess, point, with_raw_predictions=False):
        image, meta = point
        feed_dict = {
            self.image_data_placeholder: image,
            self.lat_placeholder: meta["lat"],
            self.lng_placeholder: meta["lng"],
            self.week_placeholder: meta["week"],
            "Placeholder:0": 1,
        }
        if with_raw_predictions:
            results, raw_results = sess.run([self.result_tensor, self.raw_predictions_tensor], feed_dict=feed_dict)
            return results, raw_results, self.certainty_model.get_certainty(raw_results)
        return sess.run(self.result_tensor, feed_dict=feed_dict)

    def show_random_image(self, sess, data_set):
        np.random.seed()
        point, label, id = data_set.get_random()
        prediction, raw_prediction, certainties = self.predict(sess, point, True)
        plot_image(point, prediction, data_set, raw_prediction, label, certainties)

    def show_random_images(self, sess, data_set):
        while True:
            self.show_random_image(sess, data_set)
            plt.close()

    def identify_plant(self, sess, data, data_set):
        jpeq_file_name, meta = data
        image = tf.gfile.FastGFile(jpeq_file_name, 'rb').read()
        prediction = self.predict(sess, (image, meta), with_raw_predictions=True)
        plot_image((image, meta), prediction[0], data_set, prediction[1], prediction[2])

    def save_all_results(self, data_set):
        results = []
        with tf.Session() as sess:
            for i, (image, label, identifier) in enumerate(data_set):
                print('\r>> Evaluating {} from {} {:.1f}%'.format(i + 1, data_set.size, (i + 1) / data_set.size * 100), end="")
                softmax, raw_predictions, certainties = self.predict(sess, image, with_raw_predictions=True)
                results.append({
                    "softmax": list(map(float, list(softmax[0]))),
                    "raw": list(map(float, list(raw_predictions[0]))),
                    "identifier": identifier,
                    "label": int(np.argmax(label)) if sum(label) > 0 else None,
                })
        json.dump(results, open("results-real.json", "w"))

# FC_data_set = FlowerCheckerDataSet()
FC_data_set = CustomFlowerCheckerDataSet(file_name="real_dataset.json"  )
FC_data_set.prepare_data(validation_size=0.1, test_size=0)
# FC_data_set.export_classes("classes.json")

# model = Model("Hidden layers, hidden_neuron_counts:[2048], distort:{'brightness': 0.3, 'crop': 0.5, 'epochs': 10, 'flip': True}, cut_early:True.pb")
model = Model("Hidden layers with meta, hidden_neuron_counts:[2048], hidden_meta_counts:[50, 50].pb")
# model.evaluate(FC_data_set.validation)
model.save_all_results(FC_data_set.validation)

# with tf.Session() as sess:
#     model.show_random_images(sess, FC_data_set.validation)
#     model.identify_plant(sess, ("/home/thran/kytka.jpg", defaultdict(lambda: 0)), FC_data_set)
