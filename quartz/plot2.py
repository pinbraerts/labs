import pandas as pd

ace = pd.read_csv('acetone.csv', names=['w', 'a'], skiprows=8)
air = pd.read_csv(    'air.csv', names=['w', 'a'], skiprows=8)

import matplotlib.pyplot as plt
ace.a /= ace.a.max()
air.a /= air.a.max()
plt.plot(ace.w, ace.a, label='acetone')
plt.plot(air.w, air.a, label='air')
plt.quiver(499, 1, 503 - 499, 0,
           scale=1, angles='xy', scale_units='xy')
plt.xlabel(r'$\lambda, nm$')
plt.ylabel('Reflectance')
plt.legend()
plt.savefig('pictures/reflectance.png', dpi=200)
