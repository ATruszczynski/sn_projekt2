from art2 import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
from sklearn import preprocessing
import itertools

from art2 import loader
from art2.art import Art2
from astar_art import ART2

alekPathToFolder = "C:\\Users\\aleks\\Desktop\\SN_projekt2\\SN_projekt2\\klastrowanie\\"
pathToFolder = alekPathToFolder

def codingFunc(points: np.ndarray, step: float, a: float = 10, b:float = 2):
    # dims = []
    # s = points.shape[1];
    # if s == 2:
    #     offsets = [d for d in itertools.product([0, 0.5, 1], [0, 0.5, 1])]
    # elif s == 3:
    #     offsets = [d for d in itertools.product([0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1])]



    offset = np.arange(0, 1 + step, step)
    codedPoints = []
    for p in range(points.shape[0]):
        point = points[p,:]
        codedPoint = []
        for s in range(point.shape[0]):
            x = point[s]
            for o in offset:
                codedPoint.append(np.exp(-a * np.abs(x - o) ** b))
        codedPoints.append(codedPoint)

    return np.array(codedPoints)



def hexagons():
    points, labels = loader.load_data_from_file(f"{pathToFolder}hexagon.csv")

    codedPoints = codingFunc(points, step=0.25)


    net = Art2(codedPoints.shape[1], len(np.unique(labels)), 0.95, 0.0001)
    net.learn(codedPoints, epochs=100, learning_its=5)

    plt.scatter(points[:,0], points[:,1], c=labels)
    plt.title("Original")
    plt.show()

    labels2 = []
    for i in range(len(points)):
        labels2.append(net.predict(codedPoints[i]))

    plt.scatter(points[:,0], points[:,1], c=labels2)
    plt.title("Clusterisation")
    plt.show()

    randScore = sm.adjusted_rand_score(labels, labels2)
    labels2 = np.array(labels2)
    for i in range(len(np.unique(labels2))):
        print(f'Class {i} - {len(labels2[labels2==i])}')
    print(randScore)

def cube():
    points, labels = loader.load_data_from_file(f"{pathToFolder}cube.csv")

    codedPoints = codingFunc(points=points, step=0.05, a=20, b=1.5)
    # print(codedPoints)

    # net = Art2(codedPoints.shape[1], len(np.unique(labels)), 0.95, 0.0001)
    # net.learn(codedPoints, epochs=100, learning_its=5)
    net = Art2(codedPoints.shape[1], len(np.unique(labels)), 0.90, 0.0001)
    net.learn(codedPoints, epochs=10, learning_its=5)

    labels2 = []
    for i in range(len(points)):
        labels2.append(net.predict(codedPoints[i]))


    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels)
    plt.title("Original")
    plt.show()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels2)
    plt.title("Clusterisation")
    plt.show()

    randScore = sm.adjusted_rand_score(labels, labels2)
    labels2 = np.array(labels2)
    for i in range(len(np.unique(labels2))):
        print(f'Class {i} - {len(labels2[labels2==i])}')
    print(randScore)


cube()
# dd  = itertools.product([1,2,3],['a','b'],[4,5])
# for i in dd:
#     print(i)
# d = codingFunc(np.array([[0,0], [0.5, 0.5], [1, 1]]))
# print(d)
#
# r = []
