import numpy as np


def linear(a, b):
    return lambda x: b + a * x


def mnk_err(ax, x, y, *args, **kwargs):
    (a, b), residuals, *_ = np.polyfit(x, y, 1, full=True)
    residuals = residuals[0]
    xerr = kwargs.pop('xerr', 0)
    yerr = np.sqrt(residuals / len(x)) + kwargs.pop("yerr", 0)
    #  label = f'{a:6.0} ' + kwargs.pop('label', '')
    label = kwargs.pop('label', '')
    line, *_ = ax.errorbar(x, y, linestyle="None", marker=".", yerr=yerr, label=label,
                           *args, **kwargs)
    f = linear(a, b)
    mm = min(x), max(x)
    fm = tuple(map(f, mm))
    ax.plot(mm, fm, linestyle="dashed", color=line.get_color()
            if len(label) > 1 else None)
    #  print(yerr, fm[0] - fm[1], xerr, mm[1], mm[0])
    return a, b, a * (yerr / fm[1] + xerr / mm[1])


n_air = 1
n_quartz = 1.4585
n_ace = 1.35
dn = np.pi / np.sqrt(2) / 3

crystals = [
    [
        np.array([135, 140, 146, 150]) - 110,
        np.array([595.913, 582.895, 567.605, 549.576]),
        n_air, 'crystal 1',
    ], [
        np.array([0, 255 - 150]),
        np.array([610.25, 591.426]),
        n_ace, 'crystal 1 + acetone',
    ], [
        np.array([60, 80, 90]) - 40,
        #  np.array([709.36, 715.113, 630.81, 638.131, 615.619]),
        np.array([707, 650, 615]),
        n_ace, 'crystal 2 + acetone',
        #  ], [
        #  np.array([130, 140]) - 130,
        #  np.array([709.36, 715.113, 630.81, 638.131, 615.619]),
        #  np.array([735, 709]),
        #  n_ace, 'crystal 2 + acetone',
    ], [
        np.array([250, 260, 270, 280]) - 240,
        #  np.array([570.306, 696.654, 673.067, 649]),
        np.array([602, 597, 583, 568]),
        n_ace, 'crystal 2 + acetone',
    ], [
        np.array([110, 100, 90, 80])[::-1] - 80,
        #  np.array([599.948, 694.795, 568.055, 555.342, 605.772]),
        np.array([635, 620, 593, 583]),
        n_ace, 'crystal 3 + acetone'
    ],
    [
        #  np.array([170, 180, 190, 200, 175, 185, 165, 160]) - 135,
        np.array([270, 280, 289, 290, 279, 282, 275, 270]) - 270 + 35,
        np.array([644, 628, 621.878, 608.9, 636.164, 628.131, 644.187, 647.75]),
        n_air, 'crystal 4 air'
    ],
    [
        #  np.array([175, 180, 185, 165, 160]) - 135,      #170,
        np.array([300, 305, 306, 298, 292]) - 300 + 40, #310, 
        np.array([685, 676, 670, 693, 701]),            #692, 
        n_ace, 'crystal 4 acetone'
    ],
    [
        #  np.array([160, 170, 150, 165]) - 135,
        np.array([300, 305, 297, 302]) - 300 + 40,
        np.array([605, 600, 607, 602]),
        n_air, 'crystal 1 air'
    ],
    [
        #  np.array([150, 155, 160, 170]) - 135,
        np.array([297, 302, 306, 310]) - 300 + 40,
        np.array([653, 648, 643, 637]),
        n_ace, 'crystal 1 acetone'
    ],
    [
        np.array([270, 280, 289, 290, 279, 282, 275, 270]) - 270 + 35,
        np.array([644, 628, 622, 608, 634, 628, 645, 648]),
        n_air, '4 air'
    ],
    [
        np.array([300, 305, 306, 298, 292]) - 300 + 40,
        np.array([687, 677, 671, 695, 701]),
        (n_ace + n_air) / 2, '4 acetone'
    ],
    [
        np.array([300, 305, 297, 302]) - 300 + 40,
        np.array([605, 602, 610, 603]),
        n_air, '1 air'
    ],
    [
        np.array([297, 302, 306, 310]) - 300 + 40,
        np.array([650, 648, 643, 635]),
        (n_ace + n_air) / 2, '1 acetone'
    ],
]


def plot_lam(name, phi, lam, n, ax, **kwargs):
    n_eff = np.sqrt(n * n * (1 - dn) + dn * n_quartz * n_quartz)
    phi = np.sqrt(1 - (n / n_eff) * np.sin(np.deg2rad(phi)))
    if ax == plt:
        ax.xlabel(r'$\cos(\phi)$')
        ax.ylabel(r'$\lambda, nm$')
    else:
        ax.set_xlabel(r'$\cos(\phi)$')
        ax.set_ylabel(r'$\lambda, nm$')
        ax.set_title(name)
    if len(phi) == 1:
        return
    elif len(phi) == 2:
        plt.errorbar(phi, lam, marker='+', linestyle='none', **kwargs)
        return
    elif len(phi) > 3:
        #  lam = np.delete(lam, i)
        #  phi = np.delete(phi, i)
        pass
    a, b, da = mnk_err(ax, phi, lam, **kwargs)
    d, dd = np.array([a, da]) / 2 / n_eff
    r, dr = np.array([d, dd]) * np.sqrt(8 / 3)
    print(
        f'{name:20}: n = {n:1.3f}, n_eff = {n_eff:1.3f}',
        f'd = {d:3.0f} +- {dd:2.0f} ({int(100 * dd / d):2}%), ',
        f'r = {r:3.0f} +- {dr:3.0f} ({int(100 * dr / r):2}%), ',
        f'lambda_0: {linear(a, b)(1):7.3f}'
    )


import matplotlib.pyplot as plt
#  plt.rcParams['text.usetex'] = True

oldi = 0
for phi, lam, n, name in crystals[-4:]:
    plot_lam(name, phi, lam, n, plt, label=name, xerr=0.1, yerr=5)
plt.legend()
plt.savefig('pictures/crystals.png')
plt.show()
