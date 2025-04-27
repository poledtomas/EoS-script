import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


test = np.loadtxt("/Users/tomaspolednicek/Desktop/EoS-script/EOS.dat", skiprows=1)

test_e = test[:, 0]
test_nb = test[:, 1]
test_P = test[:, 2]
test_T = test[:, 3]
test_muB = test[:, 4]

e_lin = np.linspace(test_e.min(), test_e.max(), 100)
nb_lin = np.linspace(test_nb.min(), test_nb.max(), 100)
E, NB = np.meshgrid(e_lin, nb_lin)

T_grid = griddata((test_e, test_nb), test_P, (E, NB), method="linear")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(E, NB, T_grid, cmap="viridis", edgecolor="none", alpha=0.9)

ax.set_xlabel("e")
ax.set_ylabel("nB")
ax.set_zlabel("P")


fig.colorbar(surf, ax=ax, label="P")
plt.tight_layout()
plt.show()
