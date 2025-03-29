import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata


def open_all_files(directory):
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if "inventer" in filename:
                    print(f"Opening {filename}")
                    inv = pd.read_csv(
                        filepath,
                        sep="\s+",
                        skiprows=1,
                        names=["e", "nb", "Te", "Tnb", "mub_e", "mub_nb"],
                    )

        return inv

    except Exception as e:
        print(f"Error: {e}")
        return None


directory = "/Users/tomaspolednicek/Desktop/EoS-script/data"
inv = open_all_files(directory)
e = inv["e"].to_numpy()
nb = inv["nb"].to_numpy()
Te = inv["mub_nb"].to_numpy()

E, NB = np.meshgrid(
    np.linspace(e.min(), e.max(), len(e)), np.linspace(nb.min(), nb.max(), len(e))
)

Te_grid = griddata((e, nb), Te, (E, NB), method="linear")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(E, NB, Te_grid, cmap="viridis")
# ax.scatter(e, nb, Te, c="r", marker="o", label="Data Points")
ax.set_xlabel("e")
ax.set_ylabel("nb")
ax.set_zlabel("mub_e")

plt.show()
