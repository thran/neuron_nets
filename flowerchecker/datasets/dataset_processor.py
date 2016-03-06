import json
from six.moves import urllib

import scipy.io


def procces_102_oxford_dataset():
    labels = scipy.io.loadmat("datasets/102-Oxford/imagelabels.mat")
    labels = labels["labels"][0]


def download_flowerchecker_dataset(size="big"):
    DATA_URL = "http://images.flowerchecker.com/images/{}-{}"
    FILE_PATH = "datasets/flowerchecker/images/{}.jpg"

    data = json.load(open("datasets/flowerchecker/images.json"))
    for label, d in data.items():
        for i, image in enumerate(d["images"]):
            print('\r>> Downloading FC images of {}: {} from {} - {:.1f}%'
                  .format(label, i + 1, len(d["images"]), (i + 1) / len(d["images"]) * 100.0), end="")
            urllib.request.urlretrieve(DATA_URL.format(image, size), FILE_PATH.format(image))
        print()

download_flowerchecker_dataset()