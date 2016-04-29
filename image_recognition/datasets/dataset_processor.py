import json
import os
import urllib.request
import urllib.error

import scipy.io


def procces_102_oxford_dataset():
    labels = scipy.io.loadmat("102-Oxford/imagelabels.mat")
    labels = labels["labels"][0]


def download_flowerchecker_dataset(dataset_file="flowerchecker/images.json", size="big"):
    DATA_URL = "http://images.flowerchecker.com/images/{}-{}"
    FILE_PATH = "flowerchecker/images/{}.jpg"

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

# download_flowerchecker_dataset("flowerchecker/dataset_v2_small.json")
