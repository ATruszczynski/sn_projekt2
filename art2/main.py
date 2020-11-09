from art2 import *
from random import *

net = Art2(4,2, 0.95, 0.001)


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

for i in range(10):
    new = np.zeros(4)
    c = choice([0, 1])

    new[c] = 1

    for j in [2,3]:
        new[j] = randint(0,1)

    for j in range(4):
        new[j] += normalvariate(0,0.2)
    inputs.append(new)


net.learn(inputs=inputs, epochs=5, learning_its=5)

# p = net.predict(np.array([1, 0,  0]))
# print(p)
# p = net.predict(np.array([1, 0, 0]))
# print(p)
# p = net.predict(np.array([0, 0, 1]))
# print(p)
# p = net.predict(np.array([0, 1, 0]))
# print(p)
# p = net.predict(np.array([0, 1, 0]))
# print(p)
# p = net.predict(np.array([0, 0, 1]))
# print(p)

p = net.predict(np.array([1, 0,  0, 0]))
print(p)
p = net.predict(np.array([1, 0, 1, 1]))
print(p)
p = net.predict(np.array([1, 0, 1, 0]))
print(p)
p = net.predict(np.array([0, 1, 0, 0]))
print(p)
p = net.predict(np.array([0, 1, 1, 1]))
print(p)
p = net.predict(np.array([0, 1, 1, 0]))
print(p)