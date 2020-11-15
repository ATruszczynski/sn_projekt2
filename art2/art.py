import numpy as np
import warnings as wrn

class Art2:
    def __init__(self, input_size: int, vigilance: float, theta: float):
        self.input_size = input_size
        self.vigilance = vigilance
        self.theta = theta

        self.s = np.zeros(self.input_size)
        self.f1 = np.zeros(self.input_size)
        self.f2 = np.zeros(1)
        # self.first_free = 0

        self.a = 10
        self.b = 10
        self.c = 0.1
        self.d = 0.9
        self.e = 0.00001

        self.alpha = 0.6

        self.w = np.zeros(self.input_size)
        self.x = np.zeros(self.input_size)
        self.u = np.zeros(self.input_size)
        self.v = np.zeros(self.input_size)
        self.p = np.zeros(self.input_size)
        self.q = np.zeros(self.input_size)

        self.defwei = 1.0
        self.wei = np.ones([self.input_size, 1]) * self.defwei
        self.t = np.zeros([1, self.input_size])
        
        self.added = 0

    def learn(self, inputs: [np.ndarray], epochs: int, learning_its: int):

        for epoch in range(epochs): # step 1
            for i in range(len(inputs)): # step 2

                self.s = inputs[i]
                # print(i)
                # steps 3-8
                j = self.predict(self.s, True)
                # print(j)
                for it in range(learning_its):
                    # step 9
                    # print(self.u)

                    self.t[j,:] = self.alpha * self.d * self.u + (1 + self.alpha * self.d * (self.d - 1)) * self.t[j,:]
                    self.wei[:,j] = self.alpha * self.d * self.u + (1 + self.alpha * self.d * (self.d - 1)) * self.wei[:,j]
                    # step 10

                    self.update_F1_act(j)
            print(f'epoch - {epoch}. Added {self.added} clusters')
            self.added = 0

            # step 11
            # TODO stop condition when weight aren't updated anymore?

            # step 12
            # if all epochs done, then stop

            pass

    def forward_prop(self):
        self.init_F1_act()
        # step 3

        # step 3.1
        self.u = np.zeros(self.input_size)
        self.p = np.zeros(self.input_size)
        self.q = np.zeros(self.input_size)
        self.update_W() # self.w = self.s
        self.update_X() # self.x = self.s / (self.e + self.norm(self.s))
        self.update_V() # self.v = self.threshold_func(self.x)

        # step 3.2

        self.update_F1_act(-1)

        # step 4

        self.f2 = np.dot(self.wei.T, self.p)

    def select_resonant(self, learning: bool) -> int:
        reset = True
        while reset:

            # step 6
            j = np.argmax(self.f2)
            if(self.f2[j] == -1):
                if not learning:
                    break
                j = self.add_cluster()
                self.added += 1

            # step 7
            self.update_U() # self.u = self.v / (self.e + self.norm(self.v))
            self.update_P(j) # self.p = self.u + self.d * self.t[j,:]
            r = (self.u + self.c * self.p) / (self.e + self.norm(self.u) + self.c * self.norm(self.p))

            if self.norm(r) < self.vigilance - self.e: # no resonance
                self.f2[j] = -1 # go to step 5

            else: # resonance
                # print(self.norm(r))
                self.update_W() # self.w = self.s + self.a * self.u
                self.update_X() # self.x = self.w / (self.e + self.norm(self.w))
                self.update_Q() # self.q = self.p / (self.e + self.norm(self.p))
                self.update_V() # self.v = self.threshold_func(self.x) + self.b * self.threshold_func(self.q)
                reset = False

        return j

    def predict(self, vector: np.ndarray, learning: bool = False) -> int:
        self.s = vector
        self.forward_prop()
        # print(self.p)
        j = self.select_resonant(learning)
        return j

    def add_cluster(self):
        n = self.f2.shape[0]

        self.f2 = np.append(self.f2, 1)
        self.wei = np.append(self.wei, np.ones([self.input_size, 1]) * self.defwei, axis=1)
        self.t = np.append(self.t, np.zeros([1, self.input_size]), axis=0)

        return n

    def norm(self, vector: np.ndarray):
        return np.sqrt(np.sum(np.square(vector)))

    def threshold_func(self, vector: np.ndarray):
        result = vector.copy() # TODO is this copy?
        result[np.abs(vector) < self.theta] = 0 # TODO how should it react to negative numbers?
        return result

    def update_W(self):
        self.w = self.s + self.a * self.u
        # print(f'W: {self.w}')

    def update_U(self):
        self.u = self.v / (self.e + self.norm(self.v))
        # print(f'U: {self.u}')

    def update_P(self, j: int):
        if j >= 0:
            self.p = self.u + self.d * self.t[j,:]
        else:
            self.p = self.u.copy() # TODO this may be unnecessary (when it's used t[j,:] = 0(?), so both cases are equivalent)
        # print(f'P: {self.p}')

    def update_X(self):
        self.x = self.w / (self.e + self.norm(self.w))
        # print(f'X: {self.x}')

    def update_Q(self):
        self.q = self.p / (self.e + self.norm(self.p))
        # print(f'Q: {self.q}')

    def update_V(self):
        self.v = self.threshold_func(self.x) + self.b * self.threshold_func(self.q)
        # print(f'V: {self.v}')

    def update_F1_act(self, ind:int):
        self.update_U() # self.u = self.v / (self.e + self.norm(self.v))
        self.update_W() # self.w = self.s + self.a * self.u
        self.update_P(ind) # self.p = self.u.copy()
        self.update_X() # self.x = self.w / (self.e + self.norm(self.w))
        self.update_Q() # self.q = self.p / (self.e + self.norm(self.p))
        self.update_V() # self.v = self.threshold_func(self.x) + self.b * self.threshold_func(self.q)

    def init_F1_act(self):
        self.w = np.zeros(self.input_size)
        self.x = np.zeros(self.input_size)
        self.u = np.zeros(self.input_size)
        self.v = np.zeros(self.input_size)
        self.p = np.zeros(self.input_size)
        self.q = np.zeros(self.input_size)