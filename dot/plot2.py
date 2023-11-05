import numpy as np
import matplotlib.pyplot as plt

peaks = {
    't0lumi': [606, 613, 619, 622, 629, 632, 637, 640],
    't0trans': [579, 582, 585, 588, 593, 596, 600],
    't1lumi': [522, 524, 527, 529, 532, 534, 536, 543, 546, 550, 553, ],
    't1trans': [569, 574, 579, 582, 585, 588, 593, 596, 599, 610, 616, 620, 646],
    't2lumi': [578, 582, 587, 589, 592, 597, 602, 604, 606, 610, 613, 618, 620],
    't2trans': [613, 617, 621, 624, 630, 638, 646],
    't3lumi': [606, 613, 618, 623, 632, 636, 649],
    't3trans': [579, 582, 585, 588, 593, 596, 600],
}

for k, v in peaks.items():
    plt.plot(np.arange(len(v)), v, marker='.', label=k)
plt.legend()
plt.xlabel('n')
plt.ylabel(r'$\lambda, nm$')
plt.savefig('lambdas.png')
