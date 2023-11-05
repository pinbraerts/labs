import matplotlib.pyplot as plt
import numpy as np

A = np.array([30, 29, 28, 27, 30, 32, 33, 34, 35, 36, 37])
l = 785
l1 = np.array([775, 774, 775, 773, 775, 779, 780, 781, 784, 784, 784])
l2 = np.array([790, 791, 790, 793, 790, 789, 789, 788, 787, 786, 786])
dl = l2 - l1
x = np.linspace(725, 850, 100)
gauss = lambda x, x0, s: np.exp(-((x - x0) / s) ** 2 / 2) / np.sqrt(2 * np.pi) / s
fig, axs = plt.subplots(1, 2)
for a, s in sorted(zip(A, dl)):
    axs[0].plot(x, gauss(x, l, s), label=str(a) + ' mA')
axs[0].legend()
axs[0].grid()
axs[0].set_xlabel(r'$\lambda, nm$')
dl = dl * 3e8 / 785e6
axs[1].errorbar(A, dl, xerr=1, yerr=3e2 / 785, linestyle='None', marker='+')
axs[1].set_xlabel('A, mA')
axs[1].set_ylabel(r'$\Delta \nu, THz$')
a, b = np.polyfit(A, dl, 1)
minmax = min(A), max(A)
axs[1].plot(minmax, tuple(map(lambda x: a * x + b, minmax)), label=f'{b:0.3} THz - {-a:0.3} [THz/mA] * A')
axs[1].legend()
axs[1].grid()
plt.show()

