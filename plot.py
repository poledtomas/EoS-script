import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

dir = "/Users/tomaspolednicek/Desktop/EoS-script/data/"

dataX0 = pd.read_csv(
    dir + "Press_Final_PAR_143_350_3_93_143_286_3D.dat",
    sep="\s+",
    header=None,
    # names=["e", "nb", "nQ","T", "P", "mub", "mus", "muq"],
    # names=["e", "nb", "T", "mub", "P"],
    names=["e", "nb", "P"],
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x = np.linspace(dataX0["e"].min(), dataX0["e"].max(), 100)
y = np.linspace(dataX0["nb"].min(), dataX0["nb"].max(), 100)
X, Y = np.meshgrid(x, y)
Z = griddata((dataX0["e"], dataX0["nb"]), dataX0["P"], (X, Y))

ax.plot_surface(X, Y, Z, cmap="viridis")

# ax.plot_trisurf(dataX0["e"], dataX0["nb"], dataX0["P"], cmap="viridis")
# ax.scatter(dataX0["e"], dataX0["nb"], dataX0["P"], c=dataX0["P"], cmap="viridis")

plt.show()
