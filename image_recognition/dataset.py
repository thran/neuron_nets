import json
import os
import numpy as np
import tensorflow as tf
from collections import Counter

from image_recognition.utils import hash_str, dense_to_one_hot


class DataSet:
    def __init__(self, seed=42):
        self.finished_epochs = 0
        self.class_count = 0
        self.size, self.size_train, self.size_test, self.size_validation = 0, 0, 0, 0
        self._classes = None
        self._labels = []
        self._data = []
        self._identifiers = []
        self._position = 0
        self._position_part = 0
        np.random.seed(seed)

    def prepare_data(self, validation_size=0.1, test_size=0.1, balanced_train=False):
        self._load_data()
        self._shuffle_data()
        self._split_data(validation_size=validation_size, test_size=test_size)
        if balanced_train:
            self.train._balance_data()
        self._position = 0

    def _pre_process_points(self, points, labels, identifiers):
        ps, ls, ids = [], [], []
        for point, label, identifier in zip(points, labels, identifiers):
            p, l, i = self._pre_process_point(point, label, identifier)
            ps.append(p)
            ls.append(l)
            ids.append(i)
        return ps, ls, ids

    def _pre_process_point(self, point, label, identifier):
        return point, label, identifier

    def _load_data(self):
        pass

    def _shuffle_data(self):
        r = list(range(self.size))
        np.random.shuffle(r)
        self._labels = self._labels[r]
        self._data = self._data[r]
        self._identifiers = self._identifiers[r]

    def _split_data(self, validation_size, test_size):
        validation_size = int(self.size * validation_size)
        test_size = int(self.size * test_size)
        train_size = self.size - test_size - validation_size
        train_end_position = train_size
        validation_end_position = train_size + validation_size

        self.train = self.__class__()
        self.train.size = train_size
        self.train._data = self._data[0:train_end_position]
        self.train._labels = self._labels[0:train_end_position]
        self.train._identifiers = self._identifiers[0:train_end_position]

        self.validation = self.__class__()
        self.validation.size = validation_size
        self.validation._data = self._data[train_end_position:validation_end_position]
        self.validation._labels = self._labels[train_end_position:validation_end_position]
        self.validation._identifiers = self._identifiers[train_end_position:validation_end_position]

        self.test = self.__class__()
        self.test.size = test_size
        self.test._data = self._data[validation_end_position:]
        self.test._labels = self._labels[validation_end_position:]
        self.test._identifiers = self._identifiers[validation_end_position:]

        for subset in [self.train, self.validation, self.test]:
            subset._classes = self._classes
            subset.class_count = self.class_count

    def get_class(self, label):
        if type(label) == int:
            return self._classes[label]
        else:
            return self._classes[np.argmax(label)]

    def get_batch(self, batch_size):
        if batch_size > self.size:
            raise ValueError("Batch size is larger than data set size")

        if self._position + batch_size >= self.size:
            self._shuffle_data()
            self._position = 0
            self.finished_epochs += 1
        start, end = self._position, self._position + batch_size
        self._position = end
        return self._pre_process_points(
            self._data[start:end],
            self._labels[start:end],
            self._identifiers[start:end]
        )

    def get_part(self, part_size):
        if self._position_part == self.size - 1:
            return None

        if self._position_part + part_size >= self.size:
            start, end = self._position_part, self.size - 1
        else:
            start, end = self._position_part, self._position_part + part_size
        self._position_part = end
        return self._pre_process_points(
            self._data[start:end],
            self._labels[start:end],
            self._identifiers[start:end]
        )

    def export_classes(self, file_name):
        json.dump(self._classes, open(file_name, "w"))

    def get_one(self):
        ds, ls, ids = self.get_batch(1)
        return ds[0], ls[0], ids[0]

    def get_random(self):
        position = np.random.choice(range(self.size))
        return self._pre_process_point(
            self._data[position],
            self._labels[position],
            self._identifiers[position]
        )

    def get_all(self):
        return self._pre_process_points(self._data, self._labels, self._identifiers)

    def __iter__(self):
        for _ in range(self.size):
            yield self.get_one()

    def _balance_data(self):
        labels = np.argmax(self._labels, 1)
        counts = Counter(labels)
        max_count = max(counts.values())
        indexes = np.arange(self.size)
        new_indexes = []
        for cls, count in counts.items():
            wholes, rest = max_count // count, max_count - max_count // count * count
            class_indexes = indexes[labels == cls]
            new_indexes += list(class_indexes) * wholes + list(np.random.choice(class_indexes, rest, replace=False))
        np.random.shuffle(new_indexes)
        self._data = self._data[new_indexes]
        self._labels = self._labels[new_indexes]
        self._identifiers = self._identifiers[new_indexes]
        self.size = len(self._data)


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
        for i, cls in enumerate(self._classes + [None]):
            points = data[cls] if cls is not None else sum([v for k, v in data.items() if v not in self._classes], [])
            for point in points:
                image_path = os.path.abspath(os.path.join(self.dir_name, "images", "{}.jpg".format(point["image"])))
                meta = point
                self._labels.append(i if cls is not None else None)
                self._data.append((image_path, meta))
                self._identifiers.append(hash_str(image_path))
                self.size += 1
        self._labels = dense_to_one_hot(np.array(self._labels), num_classes=self.class_count)
        self._data = np.array(self._data)
        self._identifiers = np.array(self._identifiers)

    def _prepare_classes(self, data):
        self._classes = sorted(data.keys())
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