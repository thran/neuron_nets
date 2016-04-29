import glob
import json
import os
import urllib.request
import urllib.error

import scipy.io
import shutil


def procces_102_oxford_dataset():
    labels = scipy.io.loadmat("102-Oxford/imagelabels.mat")
    labels = labels["labels"][0]


def download_flowerchecker_dataset(dataset_file="datasets/flowerchecker/images.json", size="big"):
    DATA_URL = "http://images.flowerchecker.com/images/{}-{}"
    FILE_PATH = "datasets/flowerchecker/images/{}.jpg"

    data = json.load(open(dataset_file))
    print(len(data))
    for label in sorted(data.keys()):
        d = data[label]
        images = d["images"] if "images" in d else [p["image"] for p in d]
        for i, image in enumerate(images):
            count = len(images)
            print('\r>> Downloading FC images of {}: {} from {} - {:.1f}%'
                  .format(label, i + 1, count, (i + 1) / count * 100.0), end="")
            file_name = FILE_PATH.format(image)
            if not os.path.exists(file_name):
                try:
                    urllib.request.urlretrieve(DATA_URL.format(image, size), file_name)
                except urllib.error.HTTPError:
                    print("Problem with image", image)
        print()


def move_files(source, target):
    print(target, source)
    for f in glob.glob(source + "/*.*"):
        shutil.move(f, target,)


def sort_orwens_plants(target_dir='flowerchecker/images-scrape', source_dir='flowerchecker/images-australia'):
    plants = sorted(list(os.listdir(target_dir)))
    for p in sorted(os.listdir(source_dir)):
        parts = p.split(' ')
        if len(parts) > 1:
            name = parts[0] + ' ' + parts[1]
            if name in plants:
                move_files(os.path.join(source_dir, p), os.path.join(target_dir, name))
                continue
        if parts[0] in plants:
            move_files(os.path.join(source_dir, p), os.path.join(target_dir, parts[0]))
            continue
        print('miss', p)
        return
