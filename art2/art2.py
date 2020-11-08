import numpy as np

class Art2:
    def __init__(self, input_size: int, output_size: int, vigilance: float, theta: float):
        self.input_size = input_size
        self.output_size = output_size
        self.vigilance = vigilance
        self.theta = theta

        self.s = np.zeros(self.input_size)
        self.f1 = np.zeros(self.input_size)
        self.f2 = np.zeros(self.output_size)
        self.first_free = 0

        self.a = 10
        self.b = 10
        self.c = 0.1
        self.d = 0.9
        self.e = 0.00001

        self.alpha =0.6

        self.w = np.zeros(self.input_size)
        self.x = np.zeros(self.input_size)
        self.u = np.zeros(self.input_size)
        self.v = np.zeros(self.input_size)
        self.p = np.zeros(self.input_size)
        self.q = np.zeros(self.input_size)

        self.wei = np.ones([self.input_size, self.output_size]) * 5.0
        self.t = np.zeros([self.output_size, self.input_size])

    def learn(self, inputs: [np.ndarray], epochs: int, learning_its: int):


        for epoch in range(epochs): # step 1
            for i in range(len(inputs)): # step 2
                self.w = np.zeros(self.input_size)
                self.x = np.zeros(self.input_size)
                self.u = np.zeros(self.input_size)
                self.v = np.zeros(self.input_size)
                self.p = np.zeros(self.input_size)
                self.q = np.zeros(self.input_size)


                self.s = inputs[i]
                print(self.s)
                # steps 3, 4
                self.forward_prop()

                # step 5

                reset = True

                while reset:

                    # step 6
                    j = np.argmax(self.f2)
                    print(j)
                    # TODO no reaction to lack of available new classes

                    # step 7
                    self.u = self.v / (self.e + self.norm(self.v))
                    self.p = self.u + self.d * self.t[j,:]
                    r = (self.u + self.c * self.p) / (self.e + self.norm(self.u) + self.c * self.norm(self.p))

                    if self.norm(r) < self.vigilance - self.e: # no resonance
                        self.f2[j] = -1 # go to step 5
                        pass
                    else: # resonance
                        self.update_W() # self.w = self.s + self.a * self.u
                        self.update_X() # self.x = self.w / (self.e + self.norm(self.w))
                        self.update_Q() # self.q = self.p / (self.e + self.norm(self.p))
                        self.update_V() # self.v = self.threshold_func(self.x) + self.b * self.threshold_func(self.q)
                        reset = False
                        pass

                for i in range(learning_its):
                    # step 9
                    self.t[j,:] = self.alpha * self.d * self.u + (1 + self.a * self.d * (self.d - 1)) * self.t[j,:]
                    self.wei[:,j] = self.alpha * self.d * self.u + (1 + self.a * self.d * (self.d - 1)) * self.wei[:,j]
                    print(self.wei)
                    print(self.t)
                    # step 10

                    self.update_F1_act(j)

                    # step 11
                    # TODO stop condition when weight aren't updated anymore?

                # step 12
                # if all epochs done, then stop

            pass

    def forward_prop(self):
        # step 3

        # step 3.1
        self.u = np.zeros(self.input_size)
        self.p = np.zeros(self.input_size)
        self.q = np.zeros(self.input_size)
        self.w = self.s
        self.x = self.s / (self.e + self.norm(self.s))
        self.v = self.threshold_func(self.x)

        # step 3.2

        self.update_F1_act(-1)

        # step 4

        self.f2 = np.dot(self.wei.T, self.p)

    def predict(self, vector: np.ndarray):
        self.s = vector
        self.forward_prop()
        return np.argmax(self.f2)

    def norm(self, vector: np.ndarray):
        return np.sqrt(np.sum(np.square(vector)))

    def threshold_func(self, vector: np.ndarray):
        result = vector.copy() # TODO is this copy?
        result[vector < self.theta] = 0
        return result

    def update_W(self):
        self.w = self.s + self.a * self.u

    def update_U(self):
        self.u = self.v / (self.e + self.norm(self.v))

    def update_P(self, j: int):
        if j >= 0:
            self.p = self.u + self.d * self.t[j,:]
        else:
            self.p = self.u.copy()

    def update_X(self):
        self.x = self.w / (self.e + self.norm(self.w))

    def update_Q(self):
        self.q = self.p / (self.e + self.norm(self.p))

    def update_V(self):
        self.v = self.threshold_func(self.x) + self.b * self.threshold_func(self.q)

    def update_F1_act(self, ind:int):
        self.update_U() # self.u = self.v / (self.e + self.norm(self.v))
        self.update_W() # self.w = self.s + self.a * self.u
        self.update_P(ind) # self.p = self.u.copy() # TODO is this copy?
        self.update_X() # self.x = self.w / (self.e + self.norm(self.w))
        self.update_Q() # self.q = self.p / (self.e + self.norm(self.p))
        self.update_V() # self.v = self.threshold_func(self.x) + self.b * self.threshold_func(self.q)