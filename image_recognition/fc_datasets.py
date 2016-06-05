import json
from collections import defaultdict

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

    def _add_point(self, data, label, identifier):
        self._labels.append(label)
        self._data.append(data)
        self._identifiers.append(identifier)
        self.size += 1

    def _load_data(self):
        data = json.load(open(os.path.join(self.dir_name, self.file_name)))
        data = self._prepare_classes(data)

        self.size = 0
        for i, cls in enumerate(self._classes):
            points = data[cls] if cls in data else []
            for point in points:
                image_path = point["image"]
                meta = point
                self._add_point((image_path, meta), i, hash_str(image_path))

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


def change_image_codes_to_paths(file_name, dir_name="datasets/flowerchecker"):
    data = json.load(open(os.path.join(dir_name, file_name)))
    for points in data.values():
        for point in points:
            if '/' not in point['image']:
                point['image'] = os.path.abspath(os.path.join(dir_name, "images", "{}.jpg".format(point["image"])))

    json.dump(data, open(os.path.join(dir_name, file_name), 'w'))


def add_scraped_images(file_name, dir_name="datasets/flowerchecker"):
    data = json.load(open(os.path.join(dir_name, file_name)))
    print('Points before', sum(map(len, data.values())))

    for i, cls in enumerate(sorted(data.keys())):
        scrape_dir = os.path.join(dir_name, "images-scrape", cls)
        if os.path.exists(scrape_dir):
            for image in os.listdir(scrape_dir):
                image_path = os.path.abspath(os.path.join(scrape_dir, image))
                data[cls].append({
                    "lat": None,
                    "lng": None,
                    "week": None,
                    "date": None,
                    "image": image_path,
                })

    print('Points after', sum(map(len, data.values())))
    json.dump(data, open(os.path.join(dir_name, file_name.replace('.json', '.with_scrape.json')), 'w'))


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

# change_image_codes_to_paths('dataset-1024.json')
# add_scraped_images('dataset-1024.json')
