import json
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from image_recognition.dataset import FlowerCheckerDataSet

results = json.load(open("datasets/results.json"))

FC_data_set = FlowerCheckerDataSet()
FC_data_set.prepare_data(test_size=0)


def stat():
    for i, (_, label, identifier) in enumerate(list(FC_data_set.validation)[:1000]):
        l = np.argmax(label)
        ress = results[identifier]["raw"]
        pred = np.argmax(ress)
        s = sorted(ress, reverse=True)
        m = max(ress)
        # for j, r in enumerate(results[identifier]["raw"]):
        #     if r > -10:
        #         plt.plot(r, i, ".", color=("r" if r == m else "b") if j == l else "k")
        # plt.plot(s[0], ress[l], ".", color="r" if pred == l else "k")
        plt.plot(s[0], s[1], ".", color="r" if pred == l else "k")

stat()
# plt.show()
