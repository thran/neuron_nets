import json
import os
import numpy as np
import tensorflow as tf

from dataset import DataSet
from image_recognition.old_version.utils import hash_str, dense_to_one_hot


class FlowerCheckerDataSet(DataSet):
    def __init__(self, file_name="dataset.json", dir_name="datasets/flowerchecker", distortions=0):
        super().__init__()
        self.dir_name = dir_name
        self.file_name = file_name
        self._distortions = distortions

    def _load_data(self):
        data = json.load(open(os.path.join(self.dir_name, self.file_name)))
        data = self._prepare_classes(data)

        self.size = 0
        for i, cls in enumerate(self._classes):
            points = data[cls] if cls is not "unknown" else []
            for point in points:
                image_path = os.path.abspath(os.path.join(self.dir_name, "images", "{}.jpg".format(point["image"])))
                meta = point
                self._labels.append(i)
                self._data.append((image_path, meta))
                self._identifiers.append(hash_str(image_path))
                self.size += 1
        self._labels = dense_to_one_hot(np.array(self._labels), num_classes=self.class_count)
        self._data = np.array(self._data)
        self._identifiers = np.array(self._identifiers)

    def _prepare_classes(self, data):
        self._classes = sorted(data.keys()) + ["unknown"]
        self.class_count = len(self._classes)
        return data

    def _pre_process_point(self, point, label, identifier):
        image_path, meta = point
        dist_number = self.finished_epochs % (self._distortions + 1)
        if dist_number == 0:
            return (tf.gfile.FastGFile(image_path, 'rb').read(), meta), label, identifier
        meta["distorted"] = dist_number
        return (tf.gfile.FastGFile(image_path, 'rb').read(), meta), label, identifier + "-" + str(dist_number)

    def _split_data(self, validation_size, test_size):
        super()._split_data(validation_size, test_size)
        self.train._distortions = self._distortions


class CertaintyDataSet(DataSet):
    def __init__(self, file_name="datasets/results-real.json"):
        super().__init__()
        self.file_name = file_name

    def _load_data(self):
        data = json.load(open(self.file_name))
        self.size = 0
        for point in data:
            self._labels.append([
                point["label"] == np.argsort(point['raw'])[-1],
                point["label"] == np.argsort(point['raw'])[-2],
                point["label"] == np.argsort(point['raw'])[-3],
                point["label"] in np.argsort(point['raw'])[-3:],
                point["label"] in np.argsort(point['raw'])[-5:],
                point['softmax'][point["label"]] > 0.05,
            ] if point["label"] is not None else [False] * 6)
            self._data.append(sorted(point['raw']))
            self._identifiers.append(point["identifier"])
            self.size += 1

        self._labels = np.array(self._labels)
        self._data = np.array(self._data)
        self._identifiers = np.array(self._identifiers)

# FC_data_set = FlowerCheckerDataSet(file_name="dataset-big.json")
# FC_data_set.prepare_data()