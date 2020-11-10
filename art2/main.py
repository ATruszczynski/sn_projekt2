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





# seed(1001)
#
#
#
# #points, labels = loader.load_data_from_file("C:\\Users\\aleks\\Desktop\\SN_projekt2\\SN_projekt2\\klastrowanie\\hexagon.csv")
#
centres = [[0,0], [5,5], [10,10]]
points = []
labels = []

for i in range(200):
    c = randint(0,2)
    centre = centres[c]
    new = centre.copy()
    for j in range(2):
        new[j] += normalvariate(0,0.5 + c/2)
    points.append(new)
    labels.append(c)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(points)
points = pd.DataFrame(x_scaled)
points = np.array(points)

# # print(points)
# # print(labels)
#
# #points, labels = loader.load_data_from_file("C:\\Users\\aleks\\Desktop\\SN_projekt2\\SN_projekt2\\klastrowanie\\hexagon.csv")
#
# net = Art2(points.shape[1], len(np.unique(labels)), 0.99, 0.0001)
#
# net.learn(points, epochs=20, learning_its=5)
# print(net.wei)
# print(net.t)
# print(net.wei.T - net.t)
#
# plt.scatter(points[:,0], points[:,1], c=labels)
# plt.show()
#
# labels2 = []
# for i in range(len(points)):
#     labels2.append(net.predict(points[i]))
#
# plt.scatter(points[:,0], points[:,1], c=labels2)
# plt.show()
#
# randScore = sm.adjusted_rand_score(labels, labels2)
# labels2 = np.array(labels2)
# for i in range(len(np.unique(labels2))):
#     print(f'Class {i} - {len(labels2[labels2==i])}')
# print(randScore)

net = ART2.ART2(X=points)

print(net)

plt.scatter(points[:,0], points[:,1], c=labels)
plt.show()

labels2 = net

plt.scatter(points[:,0], points[:,1], c=labels2)
plt.show()

randScore = sm.adjusted_rand_score(labels, labels2)
labels2 = np.array(labels2)
for i in range(len(np.unique(labels2))):
    print(f'Class {i} - {len(labels2[labels2==i])}')
print(randScore)


