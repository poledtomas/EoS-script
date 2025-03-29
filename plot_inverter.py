import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata


def open_all_files(directory):
    try:
        press, enerdens, bardens = None, None, None
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                print(f"Opening {filename}")
                if "Press_Final_PAR_" in filename:
                    press = pd.read_csv(
                        filepath, sep="\t", header=None, names=["mub", "T", "P"]
                    )
                elif "EnerDens_Final_" in filename:
                    enerdens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["mub", "T", "e"]
                    )
                elif "BarDens_Final_" in filename:
                    bardens = pd.read_csv(
                        filepath, sep="\t", header=None, names=["mub", "T", "nb"]
                    )
                elif "inventer" in filename:
                    inv = pd.read_csv(
                        filepath,
                        sep="\s+",
                        skiprows=1,
                        names=["e", "nb", "Te", "Tnb", "mub_e", "mub_nb"],
                    )

        if not all([press is not None, enerdens is not None, bardens is not None]):
            raise ValueError("Missing required data files!")

        return press, enerdens, bardens, inv

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None


directory = "/Users/tomaspolednicek/Desktop/EoS-script/data"  # Nastavte správnou cestu k souborům
press, enerdens, bardens, inv = open_all_files(directory)
# Convert pandas Series to NumPy arrays
e = inv["e"].to_numpy()
nb = inv["nb"].to_numpy()
Te = inv["mub_e"].to_numpy()

# Create a 2D grid for interpolation
E, NB = np.meshgrid(
    np.linspace(e.min(), e.max(), len(e)), np.linspace(nb.min(), nb.max(), 50)
)

# Interpolate Te values to match the grid
Te_grid = griddata((e, nb), Te, (E, NB), method="cubic")

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot surface
ax.plot_surface(E, NB, Te_grid, cmap="viridis")
ax.scatter(e, nb, Te, c="r", marker="o", label="Data Points")
# Labels
ax.set_xlabel("e")
ax.set_ylabel("nb")
ax.set_zlabel("Te")
ax.set_title("3D Surface Plot of e, nb, and Te")

plt.show()
