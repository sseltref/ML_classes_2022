import numpy as np
import random

class RBF:
    def __init__(self, k, beta=1.0):
        self.k = k
        self.beta = beta
        self.centers = None
        self.weights = None

    def gaussian(self, center, data_point):
        return np.exp(-self.beta*np.linalg.norm(center-data_point)**2)

    def calculate_output_matrix(self, x):
        matrix = np.zeros((len(x), self.k))
        for i, x in enumerate(x):
            for j, c in enumerate(self.centers):
                matrix[i, j] = self.gaussian(
                        c, x)
        return matrix

    def select_centers(self, y):
        c = random.sample(list(y), self.k)
        return c

    def fit(self, x, y):
        self.centers = self.select_centers(x)
        matrix = self.calculate_output_matrix(x)
        self.weights = np.dot(np.linalg.pinv(matrix), y)

    def predict(self, x):
        matrix = self.calculate_output_matrix(x)
        predictions = np.dot(matrix, self.weights)
        return predictions