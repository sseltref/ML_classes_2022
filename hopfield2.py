import numpy as np


class HopfieldNetwork():
    def train_weights(self, train_data):
        num_data = len(train_data)
        self.num_neuron = train_data[0].shape[0]

        W = np.zeros((self.num_neuron, self.num_neuron))
        p = np.sum([np.sum(t) for t in train_data]) / (num_data * self.num_neuron)

        for i in (range(num_data)):
            t = train_data[i] - p
            W += np.outer(t, t)

        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        self.W = W

    def predict(self, data, num_iter=20):
        self.num_iter = num_iter
        predicted = self.forward(data)
        return predicted

    def forward(self, init_s):
        s = init_s
        e = self.energy(s)
        for i in range(self.num_iter):
            for j in range(100):
                x = np.random.randint(0, self.num_neuron)
                s[x] = np.sign(self.W[x].T @ s)
            e_new = self.energy(s)
            if e == e_new:
                return s
            e = e_new
        return s

    def energy(self, s):
        return -0.5 * s @ self.W @ s