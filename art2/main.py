from art2 import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
from sklearn import preprocessing

from art2 import loader
from art2.art import Art2
from astar_art import ART2


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