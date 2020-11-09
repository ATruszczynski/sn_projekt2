from art2 import *
from random import *
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm

from art2 import loader
from art2.art import Art2





seed(1001)

inputs = []
# for i in range(10):
#     new = np.zeros(3)
#     c = choice([0, 1, 2])
#     if c == 0:
#         new[0] = 1
#     elif c == 1:
#         new[1] = 1
#     else:
#         new[2] = 1
#     for j in range(3):
#         new[j] += normalvariate(0,0.2)
#     inputs.append(new)

# for i in range(200):
#     new = np.zeros(4)
#     c = choice([0, 1])
#
#     new[c] = 1
#
#     for j in [2,3]:
#         new[j] = randint(0,1)
#
#     for j in range(4):
#         new[j] += normalvariate(0,0.2)
#     inputs.append(new)
#
#
# net.learn(inputs=inputs, epochs=5, learning_its=5)
#
# # p = net.predict(np.array([1, 0,  0]))
# # print(p)
# # p = net.predict(np.array([1, 0, 0]))
# # print(p)
# # p = net.predict(np.array([0, 0, 1]))
# # print(p)
# # p = net.predict(np.array([0, 1, 0]))
# # print(p)
# # p = net.predict(np.array([0, 1, 0]))
# # print(p)
# # p = net.predict(np.array([0, 0, 1]))
# # print(p)
#
# p = net.predict(np.array([1, 0,  0, 0]))
# print(p)
# p = net.predict(np.array([1, 0, 1, 1]))
# print(p)
# p = net.predict(np.array([1, 0, 1, 0]))
# print(p)
# p = net.predict(np.array([0, 1, 0, 0]))
# print(p)
# p = net.predict(np.array([0, 1, 1, 1]))
# print(p)
# p = net.predict(np.array([0, 1, 1, 0]))
# print(p)

points, labels = loader.load_data_from_file("C:\\Users\\aleks\\Desktop\\SN_projekt2\\SN_projekt2\\klastrowanie\\hexagon.csv")

# print(points)
# print(labels)

net = Art2(points.shape[1], len(np.unique(labels)), 0.999, 0.001)

net.learn(points, epochs=100, learning_its=5)

plt.scatter(points[:,0], points[:,1], c=labels)
plt.show()

labels2 = []
for i in range(len(points)):
    labels2.append(net.predict(points[i]))

plt.scatter(points[:,0], points[:,1], c=labels2)
plt.show()

randScore = sm.adjusted_rand_score(labels, labels2)
labels2 = np.array(labels2)
for i in range(len(np.unique(labels2))):
    print(f'Class {i} - {len(labels2[labels2==i])}')
print(randScore)

