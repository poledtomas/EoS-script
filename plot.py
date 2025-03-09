import matplotlib.pyplot as plt
import pandas as pd

dir = "/Users/tomaspolednicek/Desktop/EoS-script/"

dataX0 = pd.read_csv(
    dir + "EoS.dat",
    sep="\t",
    header=None,
    names=["e", "nb", "T", "mub", "P"],
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# ax.plot_trisurf(dataX0["e"], dataX0["nb"], dataX0["P"], cmap="viridis")
ax.scatter(dataX0["e"], dataX0["nb"], dataX0["T"], c=dataX0["T"], cmap="viridis")

plt.show()
