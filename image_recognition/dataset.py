import json
import os
import numpy as np
import tensorflow as tf

from image_recognition.utils import hash_str, dense_to_one_hot


class DataSet:
    def __init__(self, seed=42):
        self.finished_epochs = 0
        self.class_count = 0
        self.size, self.size_train, self.size_test, self.size_validation = 0, 0, 0, 0
        self._classes = None
        self._data = None
        self._labels = None
        self._identificators = None
        self._position = 0
        np.random.seed(seed)

    def prepare_data(self, validation_size=0.1, test_size=0.1):
        self._load_data()
        self._shuffle_data()
        self._split_data(validation_size=validation_size, test_size=test_size)
        self._position = 0

    def _pre_process_samples(self, samples):
        return [self._pre_process_sample(sample) for sample in samples]

    def _pre_process_sample(self, sample):
        return sample

    def _load_data(self):
        pass

    def _shuffle_data(self):
        r = list(range(self.size))
        np.random.shuffle(r)
        self._labels = self._labels[r]
        self._data = self._data[r]
        self._identificators = self._identificators[r]

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
        self.train._identificators = self._identificators[0:train_end_position]

        self.validation = self.__class__()
        self.validation.size = validation_size
        self.validation._data = self._data[train_end_position:validation_end_position]
        self.validation._labels = self._labels[train_end_position:validation_end_position]
        self.validation._identificators = self._identificators[train_end_position:validation_end_position]

        self.test = self.__class__()
        self.test.size = test_size
        self.test._data = self._data[validation_end_position:]
        self.test._labels = self._labels[validation_end_position:]
        self.test._identificators = self._identificators[validation_end_position:]

    def get_class(self, label):
        return self._classes[label]

    def get_batch(self, batch_size):
        if batch_size > self.size:
            raise ValueError("Batch size is larger than data set size")

        if self._position + batch_size >= self.size:
            self._shuffle_data()
            self._position = 0
            self.finished_epochs += 1
        start, end = self._position, self._position + batch_size
        self._position = end
        return (
            self._pre_process_samples(self._data[start:end]),
            self._labels[start:end],
            self._identificators[start:end]
        )

    def get_one(self):
        ds, ls, ids = self.get_batch(1)
        return ds[0], ls[0], ids[0]

    def get_all(self):
        return self._pre_process_samples(self._data), self._labels, self._identificators


class FlowerCheckerDataSet(DataSet):
    def __init__(self, dir_name="datasets/flowerchecker"):
        super().__init__()
        self.dir_name = dir_name

    def _load_data(self):
        data = json.load(open(os.path.join(self.dir_name, "images.json")))
        self._classes = sorted(data.keys())
        self.class_count = len(self._classes)

        self._labels = []
        self._data = []
        self._identificators = []
        self.size = 0
        for i, cls in enumerate(self._classes):
            for image_name in data[cls]["images"]:
                image_path = os.path.abspath(os.path.join(self.dir_name, "images", "{}.jpg".format(image_name)))
                self._labels.append(i)
                self._data.append(image_path)
                self._identificators.append(hash_str(image_path))
                self.size += 1
        self._labels = dense_to_one_hot(np.array(self._labels), num_classes=self.class_count)
        self._data = np.array(self._data)
        self._identificators = np.array(self._identificators)

    def _pre_process_sample(self, sample):
        return tf.gfile.FastGFile(sample, 'rb').read()

