#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt


def linear(a, b):
    return lambda x: b + a * x


def mnk_err(ax, x, y, *args, **kwargs):
    (a, b), residuals, *_ = np.polyfit(x, y, 1, full=True)
    residuals = np.sqrt(residuals / len(x)) + kwargs.pop("yerr", 0)
    line, *_ = ax.errorbar(x, y, linestyle="None", marker=".", yerr=residuals,
                           *args, **kwargs)
    f = linear(a, b)
    mm = min(x), max(x)
    ax.plot(mm, tuple(map(f, mm)), linestyle="dashed", color=line.get_color()
            if "label" in kwargs else None)
    return a, b


P0 = 10e5  # Pa
mmH2OtoPa = 9.80665  # mmH20/Pa
R = 8.314  # J/mol/K
NA = 6.02
T = 26.8 + 273.16  # K
L = 10  # cm
lam = 6200  # A
dlam = 0  # m/del


def plot1():
    m = np.arange(-13, 16)
    a = np.array([
        0.07, 0.40, 0.72, 0.98,
        1.30, 1.64, 1.95, 2.26,
        2.54, 2.90, 3.215, 3.55,
        3.90, 4.175, 4.54, 4.85,
        5.195, 5.545, 5.77, 6.115,
        6.48, 6.835, 7.10, 7.445,
        7.78, 8.09, 8.41, 8.745, 9.02
    ])
    fig, ax = plt.subplots()
    ax.set_xlabel("m")
    ax.set_ylabel("l, mm")
    global dlam
    dlam, _ = mnk_err(ax, m, a, xerr=0)
    dlam /= 1000
    print(dlam, ", m / del")


def plot2():
    p = np.arange(-1100, 1000, 100)
    a = np.array([
        5.57, 5.52, 5.40, 5.25, 5.19, 5.08,
        4.93, 4.76, 4.61, 4.53, 4.38, 4.175,
        4.08, 3.92, 3.82, 3.69, 3.51, 3.38,
        3.24, 3.06, 3.01
    ])
    n = a * dlam * lam / L
    print(n)
    fig, ax = plt.subplots()
    ax.set_xlabel("P, mmH2O")
    ax.set_ylabel("dn")
    n_p, n_0 = mnk_err(ax, p, n, yerr=0.005, xerr=10)
    n_p = -n_p
    print(n_p, "n / mmH2O")
    al = n_p * R * T / 2 / pi / NA / mmH2OtoPa
    n_0 = sqrt(1 + 2 * pi * NA / T / R * al * 760)
    print(al, n_0)


def plot3():
    t = np.array([
        0, 60, 90, 120,
        150, 180, 210, 240,
        270, 300, 330, 360
    ])
    a = np.array([
        11.53, 8.0, 7.62, 7.24,
        6.94, 6.62, 6.33, 6.17,
        6.03, 5.73, 5.62, 5.46
    ])
    fig, ax = plt.subplots()
    a = a * dlam * lam / L
    a0 = 4.175 * dlam * lam / L
    (k, b), residuals, *_ = np.polyfit(t, 1 / (a - a0), 1, full=True)
    residuals = np.sqrt(residuals) / len(t)
    line, *_ = ax.errorbar(t, a, yerr=residuals, xerr=15, linestyle="None")
    f = lambda x: a0 + 1 / (b + k * x)
    x = np.linspace(min(t), max(t), 100)
    y = np.fromiter(map(f, x), np.float32)
    ax.plot(x, y, linestyle="dashed")
    ax.set_xlabel("t, s")
    ax.set_ylabel("n")
    n_0 = f(0)
    print(n_0)


def main():
    plot1()
    #  plot2()
    plot3()
    #  plt.show()
    #  plt.savefig('l432.1.png')
    #  plt.savefig('l432.2.png')
    plt.savefig('l432.3.png')


if __name__ == '__main__':
    main()
