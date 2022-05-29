import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from neuralnet import Neuralnet
train_data = np.array(
    [
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]])

target_xor = np.array(
    [
        [0],
        [1],
        [1],
        [0]])

model = Neuralnet(train_data, target_xor)
model.train()


x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Y=[]
k1 = x1.tolist()
k2 = x2.tolist()
for i in k1:
    for j in k2:
        Y.append(model.forward([i, j])[0, 0])
Y = np.array(Y).reshape(100,100)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)




plt.show()