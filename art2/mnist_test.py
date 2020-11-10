from mnist import MNIST
from sklearn import preprocessing
import sklearn.metrics as sm
import numpy as np

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

# test_images = train_images
# test_labels = train_labels

howMany = 1000
# howMany = len(test_images)


test_images = test_images[0:howMany]
test_labels = test_labels[0:howMany]



net = Art2(28*28, 10, 0.8, 0.001)

net.learn(test_images, 5, 2)

conf_mat = np.zeros((10,10))

labels2 = []
for i in range(len(test_images)):
    pred = net.predict(test_images[i])
    conf_mat[test_labels[i], pred] += 1
    labels2.append(pred)
randScore = sm.adjusted_rand_score(test_labels.flatten(), labels2)
labels2 = np.array(labels2)
for i in range(len(np.unique(labels2))):
    print(f'Class {i} - {len(labels2[labels2==i])}')
print(randScore)

print(conf_mat)


