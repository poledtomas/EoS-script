import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dir = "/Users/tomaspolednicek/Desktop/EoS-script/data/"

data = pd.read_csv(
    dir + "EnerDens_Final_PAR_143_350_3_93_143_286_3D.dat",
    sep="\s+",
    header=None,
    # names=["e", "nb", "nQ","T", "P", "mub", "mus", "muq"],
    # names=["e", "nb", "T", "mub", "P"],
    names=["mub", "T", "e"],
)

data["e"] = data["e"] * (data["T"] ** 4)

pivot = data.pivot(index="mub", columns="T", values="e")
mub_values = pivot.index.values
T_values = pivot.columns.values
e_values = pivot.values

MUB, T = np.meshgrid(mub_values, T_values, indexing="ij")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(MUB, T, e_values, cmap="viridis")

ax.set_xlabel("Mub")
ax.set_ylabel("T")
plt.show()
