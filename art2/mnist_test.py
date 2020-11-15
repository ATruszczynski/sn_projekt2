from typing import Union

from mnist import MNIST
from sklearn import preprocessing
import sklearn.metrics as sm
import numpy as np

from art2.main import analyse_clustering, reduce
from art2.art import Art2

mnist_path = 'C:\\Users\\aleks\\Desktop\\MNIST'

mndata = MNIST(mnist_path)

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

train_images = np.array(train_images)
test_images = np.array(test_images)

train_labels = np.array(train_labels)
train_labels = train_labels.reshape(len(train_labels), 1)
test_labels = np.array(test_labels)
test_labels = test_labels.reshape(len(test_labels), 1)

enc = preprocessing.OneHotEncoder()

enc.fit(train_labels)
onehot_train_labels = enc.transform(train_labels).toarray()
onehot_test_labels = enc.transform(test_labels).toarray()

train_images = train_images/255
test_images = test_images/255

test_images = train_images
test_labels = train_labels

# howMany = 5000
howMany = int(len(test_images)/1)
# howMany = len(test_images)


test_images = test_images[0:howMany]
test_labels = test_labels[0:howMany]

# test_images = reduce(test_images)

# net = Art2(28*28, vigilance=0.845, theta=0.001)
#
# net.learn(test_images, epochs=5, learning_its=30)

net = Art2(28*28, vigilance=0.85, theta=0.001)

net.learn(test_images, epochs=5, learning_its=30)

conf_mat = np.zeros((10,10))

labels2 = []
for i in range(len(test_images)):
    pred = net.predict(test_images[i])
    # conf_mat[test_labels[i], pred] += 1
    labels2.append(pred)

analyse_clustering(test_labels.flatten(), labels2)




# print(conf_mat)
#
# def swapCols(matrix: np.ndarray, col1: int, col2: int):
#     matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
#     return matrix
#
# def findMaxIndex(matrix: np.ndarray) -> Union[int, int]:
#     max = -np.Inf
#     row = -1
#     col = -1
#     for r in range(matrix.shape[0]):
#         for c in range(matrix.shape[1]):
#             if matrix[r,c] > max:
#                 max = matrix[r,c]
#                 row = r
#                 col = c
#
#     return [row, col]
#
#
# def negate(matrix: np.ndarray, row: int, col: int):
#
#     for r in range(matrix.shape[0]):
#         for c in range(matrix.shape[1]):
#             if r == row or c == col:
#                 val = matrix[r, c]
#                 if val > 0:
#                     matrix[r, c] = -val
#
#     return matrix
#
# def analyse_cm(matrix: np.ndarray):
#     cols = matrix.shape[1]
#     for i in range(cols):
#         row, col = findMaxIndex(matrix)
#         if(matrix[row, col] > 0):
#             matrix = swapCols(matrix, col, row)
#             matrix = negate(matrix, row, row)
#     return np.abs(matrix)
#
#
#
# conf_mat = analyse_cm(conf_mat)
# print("___")
# print(conf_mat)
#
# print(f'accuracy: {np.sum(np.diag(conf_mat))/np.sum(conf_mat) * 100} %')

