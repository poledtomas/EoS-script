import matplotlib.pyplot as plt
import pandas as pd

dir = "/Users/tomaspolednicek/Desktop/EoS-script/data/"

dataX0 = pd.read_csv(
    dir + "BarDens_Final_PAR_143_350_3_93_143_286_3D.dat",
    sep="\t",
    header=None,
    names=["x", "y", "z"],
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot_trisurf(dataX0["x"], dataX0["y"], dataX0["z"], cmap="viridis")

plt.show()
