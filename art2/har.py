import pandas as pd
from art2.art import Art2
import matplotlib.pyplot as plt
from collections import Counter

train_path = '../UCI HAR Dataset/train/'
test_path = '../UCI HAR Dataset/test/'

data = pd.read_table(train_path+'X_train.txt', sep='\s+').to_numpy()
labels = pd.read_table(train_path+'Y_train.txt').to_numpy()
activity = pd.read_table('../UCI HAR Dataset/activity_labels.txt', sep=' ', header=None, names=('ID', 'Activity'))


net = Art2(input_size=data.shape[1], vigilance=0.9435, theta=0.001)
net.learn(data,epochs=10, learning_its=30)


labels2 = []
for i in range(len(data)):
    labels2.append(net.predict(data[i]))

print(Counter(labels2).values())

plt.hist(labels, bins=10)
plt.title("Original")
plt.show()

plt.hist(labels2, bins=10)
plt.title("Clustered by ART2")
plt.show()
