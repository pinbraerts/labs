from glob import iglob
import pandas as pd
from io import StringIO
from os.path import basename
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import numpy as np


def background(x, x0, s1, s2, h1, h2):
    x = x.copy()
    x -= x0
    return h1 * np.exp(-x / s1) + h2 * np.exp(x / s2)


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

for file in iglob('clean.csv'):
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
    # df.volt[:index] -= df.volt[index]
    # df.volt[:index] *= -1
    # df.volt[:index] += 16
    # df.volt = np.log(base + df.volt)
    i = df.volt.argmax()
    df.time -= df.time[i]
    df.volt -= df.volt[i]
    # df.volt[:i] *= -1
    df.plot('time', 'volt', ax=ax)
    continue

    f1 = (df.time < 800)
    f2 = df.time.between(6200, 7600) | (df.time > 8500)
    df.volt = np.log(df.volt.max() + 1e-5 + df.volt)

    continue
    s1, I01 = np.polyfit(df.time[f1], df.volt[f1], 1)
    s2, I02 = np.polyfit(df.time[f2], df.volt[f2], 1)
    # ax.plot(df.time[:i], df.time[:i] * s1 + I01)
    # ax.plot(df.time[i:], df.time[i:] * s2 + I02)
    y = (df.time[:i] - df.time[i]) * s1
    df.volt[:i] -= y
    y2 = df.time * s2 + I02 - 2e-2
    # df.volt -= y2
    # df.volt[:i] *= -1
    # df.volt[:i] += y
    # ax.plot(df.time, y2)
    # ax.plot(df.time, b)
    # ax.set_yscale('log')
    continue
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
    # df.plot('time', 'volt', ax=ax)
    # ax.plot(
    #     df.time[fit_range],
    #     np.exp(s * df.time[fit_range] + I0),
    #     marker='.',
    #     label='fit',
    #     linestyle='dashed',
    # )
    # ax.plot(df.time, I0 + s * df.time)
    df.volt = s * df.time + I0 - df.volt
    df.volt *= 100

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
