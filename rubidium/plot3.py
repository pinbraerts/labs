
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


def gauss(x, x0, s, h):
    x = x.copy()
    x -= x0
    x /= s
    x *= x
    # return np.exp(-x / 2)
    return np.exp(-x / 2) * h


def fitting(x, *params):
    if len(params) == 0:
        return params
    data = np.zeros_like(x)
    for p in np.array_split(params, len(params) / 3):
        data += gauss(x, *p)
    return data


c = 299792458
dl = 1
ddf = c / 4 / dl / 1000 / 1000  # MHz

Fs = np.array([
    rf'$^{{{r}}}Rb:\ F = {f} \rightarrow \{{{", ".join(map(str, g))}\}}$'
    for (r, f, g) in [
        [87, 1, [0, 1, 2]],
        [85, 2, [1, 2, 3]],
        [85, 3, [2, 3, 4]],
        [87, 2, [1, 2, 3]],
    ]
])
base = 82.50

for file in iglob('linear.csv'):
    data = open(file).read()
    file = basename(file).split('.', 1)[0]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel(r'$\nu, MHz$')
    ax.set_ylabel('A, mA')
    fig.tight_layout()
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
    time_scale = interference.time[maxima].diff().mean()
    time_scale = ddf / time_scale

    if 'CH3' not in channels:
        continue

    df = channels['CH3']
    df.time *= time_scale
    df.volt = np.log(base + df.volt)

    fit_range = (
        (df.time < 1100) |
        df.time.between(3400, 5200) |
        df.time.between(6200, 7600) |
        (df.time > 9000)
    )
    s, I0 = np.polyfit(
        df.time[fit_range],
        df.volt[fit_range],
        1,
    )
    print(s)
    continue

    df.volt = s * df.time + I0 - df.volt
    df.volt *= 100
    df.plot('time', 'volt', ax=ax)

    parameters = [
        1652, 262, 10,
        2605, 262, 17,
        5465, 265, 5,
        7845, 194, 3,
    ]
    parameters, cov = curve_fit(
        fitting,
        df.time,
        df.volt,
        parameters,
    )
    y = fitting(df.time, *parameters)
    mus = parameters[::3].copy()
    start = np.insert(mus, [0], [df.time.min()])
    start += np.roll(start, -1)
    start /= 2
    end = np.append(mus, [df.time.max()])
    end += np.roll(end, -1)
    end /= 2
    colors = ['red', 'green', 'orange', 'magenta']
    ha = 'center'
    relpos = (1, -1)
    for (mu, sigma, alpha), s, e, c, fs in zip(np.array_split(parameters, 4), start, end, colors, Fs):
        rr = df.time.between(s, e)
        ax.plot(df.time[rr], y[rr], color=c, alpha=1)
        label = '\n'.join([
            fs,
            rf'$\Delta \nu = {int(mu)}\ MHz$',
            rf'$\sigma = {int(sigma)}\ MHz$',
            rf'$s = {int(alpha)}$ %',
        ])
        # print(label)
        # print(np.diff(np.array(p).T[0]) * dddf)
        level = 15
        ax.annotate(
            label,
            (mu, alpha) if alpha < level else (mu + sigma * 4, level),
            (mu + (0 if alpha < level else sigma * 4), level),
            arrowprops=dict(
                color=c,
                arrowstyle='-',
                relpos=(0.5, 0),
                # thikness=1,
            ),
            ha='center',
            va='bottom',
        )
        # relpos = (0, 1)

    ax.legend().remove()
    ax.grid(True)
    fig.savefig(file + '.png')
    fig.savefig(file + '.svg')
plt.show()
