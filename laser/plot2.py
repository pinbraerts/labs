import numpy as np
import matplotlib.pyplot as plt


def gauss(x, mu, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-np.power((x - mu) / sigma, 2) / 2)


N = 5
n = 11000
b = 84
A = 150
x0 = np.linspace(0, n, 1000)
y0 = A * gauss(x0, n / 2, b)
fig, axs = plt.subplots(1, 2, sharey=True)

x = np.array([], dtype=float)
y = np.array([], dtype=float)
for i in range(0, N):
    drift = np.random.uniform(-b * 2, b * 2)
    x = np.append(x, x0 + i * n + drift)
    a = np.random.uniform(0.5, 1.5)
    y = np.append(y, y0 * a)

axs[0].plot(x, y, color='red', label='coherence')
M = N * N * N
y = np.zeros_like(x)
for (m, s, a) in zip(np.random.uniform(0, n * N, M), np.random.uniform(b / 4, b * 2, M), np.random.uniform(0, y0.max() * 20, M)):
    y += a * gauss(x, m, s)
axs[0].plot(x, y, color='blue', label='background')

x = np.array([], dtype=float)
y = np.array([], dtype=float)
for i in range(0, N):
    drift = 0
    x = np.append(x, x0 + i * n + drift)
    a = np.random.uniform(0.8, 1.2)
    y = np.append(y, y0 * a)

axs[1].plot(x, y, color='red', label='coherence')
M = N * N
y = np.zeros_like(x)
for (m, s, a) in zip(np.random.uniform(0, n * N, M), np.random.uniform(b / 4, b * 2, M), np.random.uniform(0, y0.max() * 20, M)):
    y += a * gauss(x, m, s)
axs[1].plot(x, y, color='blue', label='background')

axs[0].set_xlabel(r'$\nu, \mu s$')
axs[1].set_xlabel(r'$\nu, \mu s$')
axs[0].set_ylabel('V')
axs[0].grid()
axs[0].legend()
axs[1].grid()
axs[1].legend()
plt.show()
