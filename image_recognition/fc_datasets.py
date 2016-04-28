import json

import numpy as np
import os
import shutil

from dataset import DataSet
from image_recognition.utils import dense_to_one_hot, hash_str, ensure_dir_exists


class FlowerCheckerDataSet(DataSet):
    def __init__(self, file_name="dataset.json", dir_name="datasets/flowerchecker"):
        super().__init__()
        self.dir_name = dir_name
        self.file_name = file_name
        self.pre_process_image = lambda img: img

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

    def _pre_process_points(self, points, labels, identifiers):
        ps, ms, ls, ids = [], [], [], []
        for point, label, identifier in zip(points, labels, identifiers):
            p, m, l, i = self._pre_process_point(point, label, identifier)
            ps.append(p)
            ms.append(m)
            ls.append(l)
            ids.append(i)
        return ps, ms, ls, ids

    def _pre_process_point(self, point, label, identifier):
        image_path, meta = point
        return self.pre_process_image(image_path), meta, label, identifier

    def get_one(self):
        ds, ms, ls, ids = self.get_batch(1)
        return ds[0], ms[0], ls[0], ids[0]


def prepare_inception_dirs(dataset, output_dir='datasets/flowerchecker/inception'):
    while True:
        try:
            image_path, meta, label, identifier = dataset.get_one()
        except:
            break
        cls = dataset.get_class(label)
        dir = os.path.join(output_dir, cls)
        ensure_dir_exists(dir)
        shutil.copy(image_path, dir)
