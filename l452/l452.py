import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def mnk_err(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray, *args, **kwargs):
    (a, b), residuals, *_ = np.polyfit(x, y, 1, full=True)
    residuals = np.sqrt(residuals / len(x))
    line, *_ = ax.errorbar(x, y, linestyle="None", marker=".", yerr=residuals, *args, **kwargs)
    f = lambda x: b + a * x
    mm = min(x), max(x)
    ax.plot(mm, tuple(map(f, mm)), linestyle="dashed", color=line.get_color() if "label" in kwargs else None)
    return a, b


def nu(h):
    print(list(map(lambda x: (min(x), max(x)), h)))
    delta = h[0] / h[1]
    nu = (h[3] - h[2]) / (h[3] + h[2])
    nu_1 = np.sqrt(delta) / (1 + delta)
    return nu / nu_1


def plot1():
    h = np.array([
        [ -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2 ],
        [ -6, -6, -5.5, -5, -4, -3, -3, -1, 0, 1, 1 ],
        [ -3, -3, -2.5, -2, -2, -2.5, -3, -3, -3, -2, -2 ],
        [ 0, 1, 1.5, 3, 5, 6, 8, 10, 13, 15, 13 ],
    ])
    for i in h:
        i += 12
    n = nu(h)
    b = np.array([ 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20 ])
    fig, ax = plt.subplots()
    b_0 = 20
    b -= b_0
    g = np.linspace(min(b), max(b), 100)
    y = np.deg2rad(g)
    y = np.cos(y)
    ax.errorbar(b, n, linestyle="None", marker=".", xerr=1, yerr=0.1)
    ax.plot(g,      y, linestyle="dashed")
    ax.plot(g, y ** 2, linestyle="dashed")
    fig.savefig("fig1.png", dpi=200)


def plot2():
    h = np.array([
        [ 0, 0, -6, 0, -10, -5, -13, -9, -10, -10, -10, -15, -7, -10, -11, -10, -8, -10, -15, -9, -7, -8, -10, -6, -8, -13, -8 ],
        [ -4, -1, -12, -20, -6, -12, -12, -11, -12, -15, -15, -4, -4, -10, -10, 1, -1, -1, -19, -9, -9, -9, -9, -9, -9, -8, -9 ],
        [ 6, 0, -5, -13, -9, -5, -13, -8, -6, 12, 5, 8, 13, 10, 10, 5, 7, 5, -7, -10, -11, -9, -9, -6, -6, -8, -5 ],
        [ 19, 21, 14, 25+8, 12, 12, -2, 1, -2, 18, 14, 12, 18, 15, 15, 10, 12, 10, 13, 10, 17, 14, 10, 14, 11, 3, 10 ],
    ], dtype=np.float32)
    off = 24
    for i in h[1:]:
        for j, k in zip(h[0], range(len(i))):
            i[k] += off - j
    h[0] += off
    l = np.array([ 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 66, 67, 70, 75, 80, 85, 86, 87, 88, 89, 90 ])
    print(h.shape, l.shape)
    n = nu(h)
    fig, ax = plt.subplots()
    ax.plot(l, n)
    fig.savefig("fig2.png", dpi=200)


def plot3():
    h = np.array([
        [   0,   1,   1,   1,   1,   0,   1,   1,  -6,  -7, -10 ],
        [ -22, -21, -20, -19, -18, -17, -17, -15, -22, -18, -30 ],
        [  -2,  -2,  -2,  -2,  -3,  -6,  -8,  -9, -16, -16, -20 ],
        [   4,   5,   5,   7,  11,  17,  19,19+5,  18,21+3,  17 ],
    ], dtype=np.float32)
    off = 37
    for i in h[1:]:
        for j, k in zip(h[0], range(len(i))):
            i[k] += off - j
    h[0] += off
    n = nu(h)
    b_0 = 20
    b = np.array([ 120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20 ]) - b_0
    fig, ax = plt.subplots()
    g = np.linspace(min(b), max(b), 100)
    y = np.deg2rad(g)
    y = np.cos(y)
    ax.errorbar(b, n, linestyle="None", marker=".", xerr=1, yerr=0.1)
    ax.plot(g,      y, linestyle="dashed")
    ax.plot(g, y ** 2, linestyle="dashed")
    fig.savefig("fig3.png", dpi=200)


def main():
    plot1()
    plot2()
    plot3()
    plt.show()


if __name__ == '__main__':
    main()
