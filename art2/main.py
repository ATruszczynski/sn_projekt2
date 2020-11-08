from art2 import *
from random import *

net = Art2(3,2, 0.6, 0.001)


inputs = []
for i in range(10):
    new = np.zeros(3)
    c = choice([0, 1])
    if c == 0:
        new[0:2] = 1
    else:
        new[2] = 1
    inputs.append(new)


net.learn(inputs=inputs, epochs=5, learning_its=1)

p = net.predict(np.array([1, 1,  0]))
print(p)
p = net.predict(np.array([1, 1, 0]))
print(p)
p = net.predict(np.array([0, 0, 1]))
print(p)
p = net.predict(np.array([1, 1, 0]))
print(p)
print(p)
p = net.predict(np.array([0, 0, 1]))
print(p)