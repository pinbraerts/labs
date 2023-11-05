import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['text.usetex'] = True

f = 0.3 # m
lam = 646e-9 # nm
data = {
        1.50: [9.21, 8.72, 8.28, 7.80, 7.12, 6.61], # * 4 um
        1.82: [9.39, 8.81, 8.33, 7.65, 7.12, 6.52],
        1.19: [9.02, 8.69, 8.35, 7.66, 7.30, 6.98],
        4.00: [9.58, 8.90, 8.38, 7.57, 7.11, 6.37],
}
x = [-3, -2, -1, 1, 2, 3]
V = []

mins = {
        1.17: [ 7, 6, 9.5],
        1.42: [ 6, 6, 7.3],
        1.54: [ 7, 7, 8.4],
        2.23: [10, 9, 9.7],
}

# for k, v in data.items():
    # l = plt.errorbar(x, v, yerr=0.1, label="   = {:0.2} MHz".format(k), linestyle="None", marker='.')
    # b, a = np.polyfit(x, v, 1)
    # plt.plot((x[0], x[-1]), (a + b * x[0], a + b * x[-1]), color=l[0]._color, linestyle="dashed")
    # Lam = f * lam / -b / 2e-7
    # speed = Lam / k * 1000
    # print("{:0.2}: {:3.3} mm, {} mpc".format(k, Lam, int(speed)))
    # V.append(speed)
# print("mean: {} += {} mpc".format(int(np.mean(V)), int(np.std(V))))

dL = 0.57 # mm
L = []
for k, (x, y, z) in mins.items():
  Lam = 2 * 2 * z / (x + y) * dL
  L.append(Lam)
xv = [1 / v for v in mins.keys()]
fig, ax = plt.subplots()
ax.errorbar(xv, L, xerr=0.01, yerr=0.07, linestyle="None", marker=".")
a, b = np.polyfit(xv, L, 1)
ax.plot((xv[0], xv[-1]), (b + a * xv[0], b + a * xv[-1]), linestyle="dashed")
print(L, "{} mpc".format(int(a * 1000)))

# plt.legend()
# plt.show()
plt.savefig("fig1.png", dpi=200)

