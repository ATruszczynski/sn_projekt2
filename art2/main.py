from art2 import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
from sklearn import preprocessing
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

from art2 import loader
from art2.art import Art2
from astar_art import ART2

# style.use('ggplot')

def analyse_clustering(labels, labels2):
    labels2 = np.array(labels2)
    for i in np.unique(labels2):
        print(f'Class {i} - {len(labels2[labels2==i])}')
    randScore = sm.adjusted_rand_score(labels, labels2)
    print(randScore)

def reduce(images: np.ndarray):
    newImages = np.zeros([0, 14 * 14])
    newImages = []
    for i in range(images.shape[0]):
        image = images[i].reshape([28, 28])
        newImage = np.zeros([14, 14])
        for r in range(newImage.shape[0]):
            for c in range(newImage.shape[1]):
                newImage[r, c] = image[2 * r, 2 * c] + image[2 * r + 1, 2 * c] + image[2 * r, 2 * c + 1] + image[2 * r + 1, 2 * c + 1]
        # newImage = image.reshape(14,2,14,2).sum(axis=1).sum(axis=2)
        newImage = newImage / 4
        newImage = newImage.reshape([1, 14 * 14])
        # newImages = np.append(newImages, newImage, axis=0)
        newImages.append(newImage)
        if i % 1000 == 0:
            print(f'Reduced {i} points')
    newImages = np.array(newImages)
    newImages = newImages.reshape([newImages.shape[0], newImages.shape[2]])
    return  newImages

def visualise_clusterisation(org_labels, clus_labels, name):
    rows = np.max(org_labels) + 1
    cols = np.max(clus_labels) + 1

    conf_matrix = np.zeros([rows, cols])

    unclassified = []

    for i in range(len(org_labels)):
        org_label = org_labels[i]
        clus_label = clus_labels[i]
        if clus_label == -1:
            unclassified.append(org_label)
        else:
           conf_matrix[org_labels[i], clus_labels[i]] += 1

    unclassified = np.array(unclassified)

    x = []
    y = []
    s = []
    for r in range(rows):
        for c in range(cols):
            x.append(r)
            y.append(c)
            s.append(conf_matrix[r,c])

    for c in np.unique(unclassified):
        count = len(unclassified[unclassified == c])
        x.append(c)
        y.append(-1)
        s.append(count)


    s = s/np.max(s) * 350

    plt.scatter(x, y, s)
    plt.xlabel("Original")
    plt.ylabel("Clusters")
    plt.xticks(range(np.min(org_labels), np.max(org_labels) + 1, 1))
    plt.yticks(range(np.min(clus_labels), np.max(clus_labels) + 1, 1))
    plt.savefig(f'img/{name}')
    plt.show()
