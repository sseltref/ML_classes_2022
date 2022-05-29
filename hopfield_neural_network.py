from hopfield2 import HopfieldNetwork
import numpy as np

A_letter = np.array([[1, 1, 1, 1, 1],
        [1,-1,-1,-1, 1],
        [1,-1,-1,-1, 1],
        [1, 1, 1, 1, 1],
        [1,-1,-1,-1, 1],
        [1,-1,-1,-1 ,1],
        [1,-1,-1,-1, 1]])

C_letter = np.array([[1, 1, 1, 1, 1],
            [1,-1,-1,-1,-1],
            [1,-1,-1,-1,-1],
            [1,-1,-1,-1,-1],
            [1,-1,-1,-1,-1],
            [1,-1,-1,-1,-1],
            [1, 1, 1, 1, 1]])
Z_letter = np.array([[ 1, 1, 1, 1, 1],
            [-1,-1,-1,-1, 1],
            [-1,-1,-1, 1,-1],
            [-1,-1, 1,-1,-1],
            [-1, 1,-1,-1,-1],
            [ 1,-1,-1,-1,-1],
            [ 1, 1, 1, 1, 1]])
A_letter = np.reshape(A_letter, 35)
C_letter = np.reshape(C_letter, 35)
Z_letter = np.reshape(Z_letter, 35)

data=[A_letter, C_letter, Z_letter]


data_noised = np.array([[1, 1, 1, -1, 1],
        [1,-1,-1,-1, 1],
        [1,-1,-1,-1, 1],
        [1, 1, 1,-1, 1],
        [1,-1,-1,-1, 1],
        [1,-1,-1,-1,-1],
        [1,-1, 1,-1, 1]])
data_noised = np.reshape(data_noised, 35)

model = HopfieldNetwork()

model.train_weights(data)
predicted = (model.predict(data_noised))

predicted = np.reshape(predicted, (7,5))
print(predicted)