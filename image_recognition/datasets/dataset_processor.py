import json
import os
import urllib.request

import scipy.io


def procces_102_oxford_dataset():
    labels = scipy.io.loadmat("102-Oxford/imagelabels.mat")
    labels = labels["labels"][0]


def download_flowerchecker_dataset(size="big"):
    DATA_URL = "http://images.flowerchecker.com/images/{}-{}"
    FILE_PATH = "flowerchecker/images/{}.jpg"

    data = json.load(open("flowerchecker/images.json"))
    for label, d in data.items():
        for i, image in enumerate(d["images"]):
            print('\r>> Downloading FC images of {}: {} from {} - {:.1f}%'
                  .format(label, i + 1, len(d["images"]), (i + 1) / len(d["images"]) * 100.0), end="")
            file_name = FILE_PATH.format(image)
            if not os.path.exists(file_name):
                urllib.request.urlretrieve(DATA_URL.format(image, size), file_name)
        print()

download_flowerchecker_dataset()