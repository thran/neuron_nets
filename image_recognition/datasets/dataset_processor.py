import glob
import json
import os
import urllib.request
import urllib.error
import tensorflow as tf
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
            if '/' in image:
                image = image.split('/')[-1][:-4]
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


def clean_dataset(file_name, dataset_class, model_class, check_image_files=False):
    def check_image(img):
        try:
            model.pre_process_image(sess, img)
        except:
            print(img)
            shutil.move(img, "/tmp/" + img.split('/')[-1])

    if check_image_files:
        dataset = dataset_class(file_name=file_name)
        dataset.prepare_data()
        dataset.pre_process_image = check_image
        with tf.Session() as sess:
            model = model_class(dataset)
            model.build_graph()
            for i, _ in enumerate(dataset):
                print("\r>>> Step: {} / {}".format(i, dataset.size), end="")

    data = json.load(open(file_name))
    for points in data.values():
        for point in points:
            if not os.path.exists(point['image']):
                points.remove(point)
    json.dump(data, open(file_name, 'w'))