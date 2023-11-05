import matplotlib.pyplot as plt
import numpy as np

def mnk_err(ax, x, y, *args, **kwargs):
    (a, b), residuals, *_ = np.polyfit(x, y, 1, full=True)
    residuals = np.sqrt(residuals / len(x))
    ax.errorbar(x, y, linestyle="None", marker=".", yerr=residuals, *args, **kwargs)
    f = lambda x: b + a * x
    mm = min(x), max(x)
    ax.plot(mm, tuple(map(f, mm)), linestyle="dashed")
    return a, b

def plot1():
    d = np.array([ 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90 ])
    D = np.array([ 0.20, 0.30, 0.50, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50, 1.60, 1.70 ])
    fig, ax = plt.subplots()
    a, b = mnk_err(ax, d, D, xerr=0.01)
    print("D / d = ", a)
    fig.savefig("fig2.png", dpi=200)

def plot2():
    D = np.array([ 0.20, 0.25, 0.30, 0.22, 0.27, 0.18, 0.15, 0.32, 0.35, 0.37, 0.40 ])
    x = np.array([ 11.8,  9.3,  7.3, 10.6,  9.3,  9.7,  9.7,  6.3,  6.1,  6.9,  6.0 ])
    m = np.array([    5,    6,    7,    6,    7,    4,    3,    7,    8,   10,   10 ])
    x_0 = 3.0
    m_0 = 3
    X = 2 * m / x
    fig, ax = plt.subplots()
    a, b = mnk_err(ax, D, X, xerr=0.01)
    f = lambda x: (x - b) / a
    print("X / D = ", a * 10)
    print("hair:", f(2 * m_0 / x_0), "mm") # ~0.3mm
    fig.savefig("fig1.png", dpi=200)

def main():
    plot1()
    plot2()
    plt.show()

if __name__ == '__main__':
    main()

