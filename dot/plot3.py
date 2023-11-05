import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(600, 800, 1000)
y = (x - 600) * np.exp(-x / 50)
plt.plot(x, y)
plt.show()
