import pandas as pd
from glob import iglob
import matplotlib.pyplot as plt
from os.path import basename

names = ['w', 'a']
a1 = plt.subplots(1, 2, figsize=[12, 5], sharey=True)
a2 = plt.subplots(1, 2, figsize=[12, 5], sharey=True)
figs = {
    '1 air': (a1[0], a1[1][0]),
    '1 acetone': (a1[0], a1[1][1]),
    '4 air': (a2[0], a2[1][0]),
    '4 acetone': (a2[0], a2[1][1]),
}
nms = {
    '4 air': dict(zip(
        [270, 280, 289, 290, 279, 282, 275, 270],
        [644, 634, 622, 608, 637, 628, 645, 648],
    )),
    '4 acetone': dict(zip(
        [300, 305, 306, 298, 292],
        [687, 677, 671, 695, 701],
    )),
    '1 air': dict(zip(
        [300, 305, 297, 302],
        [605, 602, 610, 603],
    )),
    '1 acetone': dict(zip(
        [297, 302, 306, 310],
        [650, 648, 643, 635],
    )),
}
boundaries = {
    '1 air':     [560, 660],
    '1 acetone': [620, 700],
    '4 air':     [580, 675],
    '4 acetone': [640, 740],
}

for file in iglob('zamyatin-shirshov__\\???_?_a*.txt'):
    angle, crystal, fill = basename(file).split('.')[0].split('_')
    if ' ' in fill:
        continue
    name = f'{crystal} {fill}'
    fig, axis = figs[name]
    df = pd.read_csv(file, decimal=',', sep='\t', skiprows=14, names=names)
    df = df[df.w.between(*boundaries[name])]
    df = df.rolling(window=20).mean()
    df.a -= df.a.iloc[-1]
    df.a /= df.a.max()
    (line,) = axis.plot(df.w, df.a, label=f'{angle}°')
    #  fig.savefig(file.replace('.txt', '.png'))
    w = nms[name][int(angle)]
    #  axis.axvline(w)
    axis.plot([w, w], [0, df.a.iloc[df.w.subtract(w).abs().argmin()]],
              color=line.get_color())

for k, (fig, axis) in figs.items():
    axis.set_title('crystal ' + k)
    axis.set_xlabel(r'$\lambda, nm$')
    axis.grid()
    axis.legend()
    fig.savefig(f'pictures/crystal_{k[0]}.png')

plt.show()
