import json

import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
import numpy as np
import os
import tfdeploy as td
import tensorflow as tf

from image_recognition.fc_datasets import FlowerCheckerDataSet

MODEL_DIR = "models"
IMAGE_TENSOR_NAME = 'pre_process_image/convert_image:0'
RESULT_TENSOR_NAME = 'results/predictions:0'
LAT_TENSOR_NAME = 'lat_placeholder:0'
LNG_TENSOR_NAME = 'lng_placeholder:0'
WEEK_TENSOR_NAME = 'week_placeholder:0'


class CertaintyModel:
    def __init__(self):
        self.model = td.Model(os.path.join(MODEL_DIR, "certainty_model-1025.pkl"))
        self.input, self.output = self.model.get("row_predictions", "output/certainties")

    def get_certainty(self, input):
        return self.output.eval({self.input: sorted(input[0])})


def crop_image(image, crop):
    height, width, _ = image.shape

    if crop == 'no_crop':
        pass
    if crop.startswith('left'):
        if height >= width:
            image = image[:width, :, :]
        else:
            image = image[:, :height, :]
    if crop.startswith('center'):
        if height >= width:
            image = image[(height - width)/2:(height + width)/2, :, :]
        else:
            image = image[:, (width - height)/2:(height + width)/2, :]
    if crop.startswith('right'):
        if height >= width:
            image = image[-width:, :, :]
        else:
            image = image[:, -height:, :]

    height, width, _ = image.shape
    if crop.endswith('_center'):
        image = image[height * .15:height * .85, height * .15:height * .85, :]
    if crop.endswith('_top_left'):
        image = image[:height * .7, :height * .7, :]
    if crop.endswith('_top_right'):
        image = image[:height * .7, height * .3:, :]
    if crop.endswith('_bottom_left'):
        image = image[height * .3:, :height * .7, :]
    if crop.endswith('_bottom_right'):
        image = image[height * .3:, height * .3:, :]

    return image


class Model:
    def __init__(self, model_name, classes):
        self.certainty_model = CertaintyModel()
        self._classes = classes
        with tf.Session() as sess:
            model_filename = os.path.join(MODEL_DIR, model_name)
            with tf.gfile.FastGFile(model_filename, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')
            self.graph = sess.graph
            self.result_tensor = self.graph.get_tensor_by_name(RESULT_TENSOR_NAME)
            self.raw_predictions_tensor = self.result_tensor.op.inputs[0]
            self.image = self.graph.get_tensor_by_name(IMAGE_TENSOR_NAME)
            self.lat_placeholder = self.graph.get_tensor_by_name(LAT_TENSOR_NAME)
            self.lng_placeholder = self.graph.get_tensor_by_name(LNG_TENSOR_NAME)
            self.week_placeholder = self.graph.get_tensor_by_name(WEEK_TENSOR_NAME)

            self.jpeg = tf.placeholder(dtype='string')
            image = tf.image.decode_jpeg(self.jpeg, channels=3)
            self.decoded_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            
        # self.crops = ['no_crop', 'left', 'center', 'right', 'center_center']
        self.crops = [
            'no_crop', 'left', 'center', 'right', 'center_center',
            'center_top_left', 'center_top_right', 'center_bottom_left', 'center_bottom_right'
            # 'left_top_left', 'left_top_right', 'left_bottom_left', 'left_bottom_right'
            # 'right_top_left', 'right_top_right', 'right_bottom_left', 'right_bottom_right'
        ]

    def evaluate(self, data_set):
        results = pd.DataFrame(columns=['id', 'prediction', 'true_class', 'class_known', 'genus_correct', 'correct'])
        with tf.Session() as sess:
            hits = defaultdict(lambda: 0)
            for i, (image, meta, label, identifier) in enumerate(data_set):
                print('\r>> Evaluating {} from {} ({:.1f}%) - {}'.format(
                    i + 1, data_set.size, (i + 1) / data_set.size * 100, hits), end="")
                
                true_class = data_set.get_class(label)
                with tf.gfile.FastGFile(image, 'rb') as image_file:
                    image = sess.run(self.decoded_image, {self.jpeg: image_file.read()})
                    
                    result_mean = np.zeros(len(self._classes))
                    for crop in self.crops:
                        cropped = crop_image(image, crop)
                        result, raw = self.predict(sess, (cropped, meta))
                        result_mean += raw[0]

                        predicted_class = self._classes[np.argmax(result)]
                        hits[crop] += 1 if predicted_class == true_class else 0

                    result_mean /= len(self.crops)
                    result_mean = sess.run(tf.nn.softmax(np.expand_dims(result_mean, 0)))[0]
                    predicted_class = self._classes[np.argmax(result_mean)]
                    hits['mean'] += 1 if predicted_class == true_class else 0

                    results.loc[len(results)] = [
                        identifier,
                        predicted_class,
                        true_class,
                        true_class in self._classes,
                        true_class.split(' ')[0] == predicted_class.split(' ')[0],
                        true_class == predicted_class,
                    ]
        return results

    def predict(self, sess, point):
        image, meta = point
        feed_dict = {
            self.image: image,
            self.lat_placeholder: meta["lat"],
            self.lng_placeholder: meta["lng"],
            self.week_placeholder: meta["week"],
        }
        return sess.run([self.result_tensor, self.raw_predictions_tensor], feed_dict=feed_dict)


model = Model("IncMod v0.2 - 1025 plants.pb", json.load(open('models/classes-1025.json')))
ds = FlowerCheckerDataSet(file_name='real_dataset_v3.json')
ds.prepare_data()

model.evaluate(ds).to_pickle('eval/eval_v3-{}.pd'.format('1025crops9'))

# >> Evaluating 1103 from 20460 (5.4%) - defaultdict(<function Model.evaluate.<locals>.<lambda> at 0x7fd6cdcb3f28>, {'right': 383, 'center_center': 407, 'left': 382, 'mean': 428, 'no_crop': 395, 'center': 407})