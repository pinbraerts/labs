import matplotlib.pyplot as plt
import numpy as np


def plot1():
    phi1 = np.deg2rad(np.array([36, 134, 208, 290]))
    phi2 = np.deg2rad(np.array([70, 160, 250, 340]))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.bar(phi1, np.full_like(phi1, 1), label='max', linestyle="None", linewidth=0.1, width=0.06)#, marker='.')
    ax.bar(phi2, np.full_like(phi2, 1), label='min', linestyle="None", linewidth=0.1, width=0.06)#, marker='.')
    ax.set_yticks([])
    ax.legend()


def main():
    plot1()
    plt.show()
    #  plt.savefig("l473.png")


if __name__ == '__main__':
    main()
