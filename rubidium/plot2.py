
from glob import iglob
import pandas as pd
from io import StringIO
from os.path import basename
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np


def nearest(array, point):
    return np.abs(array - point).argmin()

def gauss(x, x0, s):
    x = x.copy()
    x -= x0
    x /= s
    x *= x
    # return np.exp(-x / 2)
    return np.exp(-x / 2)


def background(x, s):
    return np.exp(x / s)


def fitting(x, b, h1, s1, r2, s2, h2, r3, s3, h3, r4, s4, h4, r5, s5, h5):
    return (
        background(x, s1) * h1 +
        gauss(x, r2, s2) * h2 +
        gauss(x, r3, s3) * h3 +
        gauss(x, r4, s4) * h4 +
        gauss(x, r5, s5) * h5 +
        b
    )


c = 299792458
dl = 1
ddf = c / 4 / dl / 1000 / 1000  # MHz

for file in iglob('linear.csv'):
    data = open(file).read()
    file = basename(file).split('.', 1)[0]
    fig, ax = plt.subplots(figsize=(14, 10))
    dddf = 0
    channels = {}
    for sheet in data.split('\n\n\n'):
        channel, sz, *_ = sheet.split('\n', 3)
        if 'OFF' in channel or '2' in channel:
            continue
        channel = channel.split(':')[-1]
        df = pd.read_csv(
            StringIO(sheet),
            skiprows=3,
            decimal='.',
            dtype=float,
            names=['index', 'time', 'volt'],
        )
        df = df.rolling(10).sum().dropna()
        channels[channel] = df

    interference = channels['CH1']
    maxima, properties = find_peaks(interference.volt, distance=25)
    maxima = maxima[1:]
    # maxima += interference.index[0]
    time_scale = interference.time[maxima].diff().mean()
    dddf = ddf / time_scale

    if 'CH3' not in channels:
        continue

    spectrum = channels['CH3']
    # interference.volt -= interference.volt.min()
    # interference.volt += spectrum.volt.max()
    # # interference.plot('time', 'volt', ax=ax, alpha=0.5)
    # maxima = interference.iloc[maxima]
    # maxima.plot(
    #     'time',
    #     'volt',
    #     ax=ax,
    #     linestyle='none',
    #     marker='.',
    #     # label='maxima',
    # )
    # if 'baseline' in file:
    #     spectrum.plot('time', 'volt', ax=ax)
    #     print(spectrum.volt.mean())
    #     ax.set_xlabel('t, s')
    #     ax.set_ylabel('A, mA')
    #     # plt.show()
    #     fig.savefig(file + '.png')
    #     fig.savefig(file + '.svg')
    #     continue

    Fs = [
        [2, [1, 2, 3]],
        [1, [0, 1, 2]],
        [2, [1, 2, 3]],
        [3, [2, 3, 4]],
    ]
    base = -82.50
    channel = 'CH1'
    df = spectrum
    radius = 0.045
    params = [
        df.volt.min() + base,
        df.volt.max() - df.volt.min() - base,
        -radius,
        # 300000,
        5e-3, 1e-3, -6e-3,
        9e-3, 1e-3, -8e-3,
        2e-2, 1e-3, -1e-3,
        2.5e-2, 1e-4, -5e-4,
    ]
    bbb = (-np.abs(df.volt.max()), 0)
    fit_range = (
        (df.time < 3e-3) |
        df.time.between(1.2e-2, 1.6e-2) |
        df.time.between(2.1e-2, 2.5e-2) |
        (df.time > 2.9e-2)
    )
    if True:
        bounds = np.array((
            (df.volt.min() + 2 * base, 0),
            (0, df.volt.max() - 2 * base - df.volt.min()),
            (-1, 0),
            # (100000, 1000000),
            (3e-3, 7e-3), (1e-4, 2e-3), bbb,
            (8e-3, 1e-2), (1e-4, 2e-3), bbb,
            (1.7e-2, 2e-2), (1e-4, 2e-3), bbb,
            (2.3e-2, 2.7e-2), (1e-4, 1e-3), bbb,
        )).T
        params, cov, info, message, error = curve_fit(
            fitting,
            df.time,#[fit_range],
            df.volt,#[fit_range],
            params,
            bounds=bounds,
            full_output=True,
        )
        if np.isnan(params.sum()):
            print('curve not converged')
            continue
    b = params[:4]
    p = np.array_split(params[3:], 4)

    back = background(df.time, b[2]) * b[1]# + b[0]
    back = back.to_numpy()
    # ax.plot(df.time, back, color='black', linestyle='dashed')
    ax.plot(df.time, (df.volt - b[0]) / back, color='grey')
    y = fitting(df.time, *params)
    ppp = np.zeros(len(p) + 2)
    ppp[1:1 + len(p)] = np.array(p).T[0]
    ppp[0] = df.time.min()
    ppp[-1] = df.time.max()
    ppp += np.roll(ppp, -1)
    ppp /= 2
    colors = ['red', 'green', 'orange', 'blue']
    for i, color in enumerate(colors):
        rr = df.time.between(ppp[i], ppp[i + 1])
        pp = p[i].copy()
        # pp[0] -= p[0][0]
        pp[:2] *= dddf
        bbbb = background(p[i][0], b[2]) * b[1] + b[0] - base
        pp[2] /= bbbb
        pp[2] *= 100
        # pp[0] += 228115.20
        ax.plot(
            df.time[rr],
            (y[rr] - b[0]) / back[rr],
            color=color,
        )
        label = rf'$\nu = {int(pp[0])}$' + ' MHz\n' + \
                rf'$\sigma = {int(pp[1])}$' + ' MHz\n' + \
                rf'$s = {-int(pp[2])}$ %' + ' MHz\n' + \
                rf'$F = {Fs[i][0]} \rightarrow \{{{", ".join(map(str, Fs[i][1]))}\}}$'
        print(label)
        print(np.diff(np.array(p).T[0]) * dddf)
        ax.annotate(
            label,
            (p[i][0], df.volt.iloc[nearest(df.time.to_numpy(), p[i][0])]),
            (p[i][0], df.volt.max()),
            arrowprops=dict(
                color=color,
                arrowstyle='-',
                relpos=(0, 1) if i > 0 else (1, 1),
                # thikness=1,
            ),
            ha='left' if i > 0 else 'right',
            va='center',
        )
        # ax.plot(df.time, y, label='fit')
    ax.legend().remove()
    ax.set_xlabel('t, s')
    ax.set_ylabel('A, mA')
    plt.show()
    # fig.savefig(file + '.png')
    # fig.savefig(file + '.svg')
