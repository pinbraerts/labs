import matplotlib.pyplot as plt
import numpy as np


def mnk_err(ax, x, y, *args, **kwargs):
    (a, b), residuals, *_ = np.polyfit(x, y, 1, full=True)
    residuals = np.sqrt(residuals / len(x)) + kwargs.pop("yerr", 0)
    line, *_ = ax.errorbar(x, y, linestyle="None", marker=".", yerr=residuals,
                           *args, **kwargs)
    mm = min(x), max(x)
    ax.plot(mm, tuple(map(lambda x: b + a * x, mm)),
            linestyle="dashed", color=line.get_color()
            if "label" in kwargs else None)
    return a, b


l = 3.5  # см
L = 89  # см
n0 = 2.29
lam = 630e-6  # м


def plot1():
    x = np.arange(1, 6)
    y = np.array([4.0, 5.5, 6.5, 7.25, 8.0])
    fig, ax = plt.subplots()
    ax.set_xlabel("m")
    ax.set_ylabel("r, см")
    y *= y
    a, b = mnk_err(ax, x, y)
    nn = a * l / lam / (n0 * L) ** 2
    print(nn)


def plot2():
    y = np.array([0, 32, 66, 94])
    x = np.arange(len(y))
    fig, ax = plt.subplots()
    ax.set_xlabel("m")
    ax.set_ylabel("U, дел")
    _ = mnk_err(ax, x, y)


def main():
    plot1()
    #  plot2()
    plt.savefig("l472.png")


if __name__ == '__main__':
    main()
