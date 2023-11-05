from glob import glob
import pandas as pd
from os.path import basename, getctime
import matplotlib.pyplot as plt
cmap = plt.get_cmap('Spectral')

values = {
    (0, 135): 595.913,
    (0, 140): 582.895,
    (0, 146): 567.605,
    (0, 150): 549.576,
    (0, 169): 545.514,
    (0, 190): 591.125,
    (0, 255): 549.426,

    (1, 140): 709,
    (1, 60): 707,
    (1, 80): 628,
    (1, 90): 615,
    #  709.36, 715.113, 630.81, 628.131, 615.619,
    #  555.342, 550, 524, 546, 540,
    (2, 90): 555,
    (2, 100): 568,
    (2, 110): 599,
    (2, 240): 605,
    (2, 390): 694,
    #  555.342, 599.948, 694.795, 605.772, 568.055,
    #  610.25, 696.654, 673.067, 649
    (3, 250): 602,
    (3, 260): 597,
    (3, 270): 583,
    (3, 280): 568,
}
names = ['freq', 'amp']
files = glob('010923Кристалл/*')
files.sort(key=getctime)
j = 0
oldi = 0
for name in files:
    df = pd.read_csv(open(name), decimal=',', sep='\t', skiprows=14, names=names)
    name = basename(name)
    i = ord(name[8]) - ord('1')
    if i != oldi:
        plt.savefig('pictures/spectre{}.png'.format(i))
        oldi = i
        j = 0
    df = df[df['freq'].between(480, 800 if i > 0 else 650)]
    df = df.rolling(30, center=True, on='freq').sum()
    df['amp'] = df['amp'] / df['amp'].max()
    plt.figure(i)
    #  plt.title(name[8])
    plt.ylabel('amplitude')
    plt.xlabel('wavelength, nm')
    angle = int(name[10:-4])
    freq = values[(i, angle)]
    angle = str(angle) + '°'
    index = df['freq'].subtract(freq).abs().argmin()
    amp = df['amp'].iloc[index]
    plt.annotate('{}nm'.format(freq), (freq, j * 0.05))
    j += 1
    line, *_ = plt.plot(
        df['freq'], df['amp'], label=angle, marker=',',
        #  color=cmap((freq - 380) / 750),
    )
    plt.plot([freq, freq], [0, amp], color=line.get_color())
    plt.legend()
plt.show()
#  plt.savefig('pictures/spectre4.png')
