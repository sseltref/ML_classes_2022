import numpy as np
import matplotlib.pyplot as plt
from RBF import RBF


x = np.linspace(-1, 1, 50)
y = np.sin((x+0.5)**3)
y_noised = y + np.random.uniform(-0.1, 0.1, x.shape)

model = RBF(k=1)

model.fit(x, y_noised)
predicted_y = model.predict(x)

plt.plot(x, y_noised, label="noised function")
plt.plot(x, predicted_y, label="predicted function")
plt.plot(x, y, label="actual function")
plt.legend(loc='upper left')
plt.show()