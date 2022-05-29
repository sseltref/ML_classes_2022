import numpy as np

class Neuralnet:
    def __init__(self, train_data, target, lr=0.01, cmax=10000, num_input=2, num_hidden=2, num_output=1, emax= 0.1):
        self.train_data = train_data
        self.target = target
        self.lr = lr
        self.cmax = cmax
        self.emax = emax

        self.weights_01 = np.random.uniform(-1, 1, size=(num_input, num_hidden))
        self.weights_12 = np.random.uniform(-1, 1, size=(num_hidden, num_output))

        self.b01 = np.random.uniform(-1, 1, size=(1, num_hidden))
        self.b12 = np.random.uniform(-1, 1, size=(1, num_output))


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _delsigmoid(self, x):
        return x * (1 - x)

    def forward(self, batch):
        self.hidden_ = np.dot(batch, self.weights_01) + self.b01
        self.hidden_out = self._sigmoid(self.hidden_)

        self.output_ = np.dot(self.hidden_out, self.weights_12) + self.b12
        self.output_final = self._sigmoid(self.output_)

        return self.output_final

    def update_weights(self):
        loss = 0.5 * (self.target - self.output_final) ** 2
        self.E = np.sum(loss)
        error_term = (self.target - self.output_final)
        grad01 = self.train_data.T @ (
                ((error_term * self._delsigmoid(self.output_final)).dot(self.weights_12.T)) * self._delsigmoid(
            self.hidden_out))

        grad12 = self.hidden_out.T @ (error_term * self._delsigmoid(self.output_final))
        self.weights_01 += self.lr * grad01
        self.weights_12 += self.lr * grad12
        self.b01 += np.sum(
            self.lr * ((error_term * self._delsigmoid(self.output_final)).dot(self.weights_12.T)) * self._delsigmoid(
                self.hidden_out), axis=0)
        self.b12 += np.sum(self.lr * error_term * self._delsigmoid(self.output_final), axis=0)

    def train(self):
        for i in range(self.cmax):
            self.E = 0
            self.forward(self.train_data)
            self.update_weights()
            assert len(self.train_data) == len(self.target)
            p = np.random.permutation(len(self.train_data))
            self.train_data, self.target = self.train_data[p], self.target[p]
            #print(self.E)
            if self.E < self.emax:
                break


