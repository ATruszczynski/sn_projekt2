from art2 import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pandas as pd
from sklearn import preprocessing
import itertools
from art2.main import analyse_clustering, visualise_clusterisation

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

def codingFunc2(points: np.ndarray):
    codedPoints = np.zeros([points.shape[0], 2 * points.shape[1]])
    dim = points.shape[1]
    for p in range(points.shape[0]):
        point = points[p,:]
        codedPoint = np.zeros([1, 2 * dim])
        for d in range(dim):
            if point[d] > 0:
                codedPoint[0, 2*d] = point[d]
            else:
                codedPoint[0, 2*d + 1] = -point[d]
        codedPoints[p, :] = codedPoint
    return codedPoints



def hexagons():
    points, labels = loader.load_data_from_file(f"{pathToFolder}hexagon.csv", norm=False) # 94%

    net = Art2(points.shape[1], 0.93, 0.01)
    net.learn(points, epochs=10, learning_its=10)

    plt.scatter(points[:,0], points[:,1], c=labels)
    plt.title("Original")
    plt.savefig('img/hexa_org')
    plt.show()

    labels2 = []
    for i in range(len(points)):
        labels2.append(net.predict(points[i]))

    plt.scatter(points[:,0], points[:,1], c=labels2)
    plt.title("Clustering")
    plt.savefig('img/hexa_clus')
    plt.show()

    analyse_clustering(labels, labels2)

    visualise_clusterisation(labels, labels2, "hexa_clus_vis")

def cube() -> Art2:
    points, labels = loader.load_data_from_file(f"{pathToFolder}cube.csv")

    points = points - np.average(points, axis=0)

    net = Art2(points.shape[1], vigilance=0.92, theta=0.005)
    net.learn(points, epochs=10, learning_its=35)

    labels2 = []
    for i in range(len(points)):
        labels2.append(net.predict(points[i]))


    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels)
    plt.title("Original")
    plt.savefig('img/cube_org')
    plt.show()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels2)
    plt.title("Clustering")
    plt.savefig('img/cube_clus')
    plt.show()

    analyse_clustering(labels, labels2)

    visualise_clusterisation(labels, labels2, "cube_clus_vis")

    return net

def cube_nm(net: Art2):
    cp, _ = loader.load_data_from_file(f"{pathToFolder}cube.csv")
    points, labels = loader.load_data_from_file(f"{pathToFolder}cube-notmatching.csv")

    points = points - np.average(cp, axis=0)

    labels2 = []
    for i in range(len(points)):
        labels2.append(net.predict(points[i]))


    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels)
    plt.title("Original")
    plt.savefig('img/cube_nm_org')
    plt.show()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(points[:,0], points[:,1], points[:,2], c=labels2)
    plt.title("Clustering")
    plt.savefig('img/cube_nm_clus')
    plt.show()

    analyse_clustering(labels, labels2)
    visualise_clusterisation(labels, labels2, "cube_nm_clus_vis")


hexagons()
net = cube()
cube_nm(net)
