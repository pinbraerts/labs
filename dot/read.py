import pandas as pd
import numpy as np
from glob import iglob
from os.path import basename
from collections import defaultdict
from scipy.optimize import curve_fit


def read_file(file, row0=518, row1=3569, w0=370, w1=705, w2=470, scale=True, norm=True, window=10):
    i = 1
    k = 0
    with open(file) as f:
        j = 0
        for line in f:
            if j < 3:
                i += 1
                j += 'clipbox' in line
            elif 'stroke' in line:
                break
            k += 1

    fd = pd.read_csv(
        file,
        delimiter=' ',
        decimal='.',
        names=['w', 'a', 'l'],
        skiprows=i,
        nrows=k - i,
    )

    if scale:
        fd.w -= fd.w.min()
        fd.w /= fd.w.max()
        fd.w *= (w1 - w0)
        fd.w += w0

    if norm:
        fd.a -= fd.a.min()
        fd.a /= fd.a.max()
        fd = fd[fd.w > w2]

    if window:
        fd.a = fd.a.rolling(window=window).mean()
    return fd


height = {
    'background': 16000,
    't0lumi': 16000,
    't0trans': 14000,
    't1lumi': 32000,
    't1trans': 15000,
    't2lumi': 27500,
    't2trans': 4500,
    't3lumi': 17000,
    't3trans': 15000,
}

split = {
    't1trans': 545,
    't2trans': 600,
    't3trans': 545,
    't0trans': 545,
}
maxm = {
    't1trans': 590,
    't2trans': 625,
    't3trans': 590,
    't0trans': 590,
}

background = read_file('zamyatin-shirshov\\background.eps', window=50)
background.a *= height['background']

import matplotlib.pyplot as plt
data = defaultdict(dict)
for name in iglob('zamyatin-shirshov\\t[0123]*.eps'):
    fig, ax = plt.subplots(sharey=True)
    fd = read_file(name)
    name = basename(name)[:-4]
    #  fd.a -= background.a
    fd.a *= height[name]
    if 'trans' in name:
        def curve(x, a, b, c):
            return height[name] * ((x - a) / c / b) ** c * np.exp(-(x - a + c * b) / b)
        a = split[name]
        b = maxm[name] - a
        i = fd.w > split[name]
        x = fd.w[i]
        par = [a, b, 2]
        par, cov = curve_fit(curve, x, fd.a[i], par)
        y = curve(x, *par)
        #  params = height[name]
        #  par, cov = curve_fit(curve, fd.w[i], fd.a[i])
        ax.plot(x, y)
    i = ord(name[1]) - ord('0')
    i %= 3
    method = name[2:]
    ax.plot(fd.w, fd.a, label=name)
    #  ax.plot(background.w, background.a, label='background')
    ax.set_xlabel(r'$\lambda, nm$')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.set_title(i)
plt.show()
#  plt.savefig('figure.png')
