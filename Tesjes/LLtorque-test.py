import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from mumax5 import Ferromagnet, Grid, World

def ana(timepoints):
    gamma = 1.7595e11
    return ([np.cos(0.1 * gamma * t * 1e-9) for t in timepoints])

def cosine(time, w):
    return np.cos(w * time * 1e-9)

length, width, thickness = 1e-9, 1e-9, 1e-9
nx, ny, nz = 1, 1, 1
world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)), 3)
magnet.msat = 800e3
magnet.aex = 5e-12

magnet.magnetization = (1, 0, 0)
world.bias_magnetic_field = (0, 0, 0.1)

timepoints = np.linspace(0, 3e-9, 75)
outputquantities = {"mx1": lambda: magnet.magnetization.average()[0]}
output = world.timesolver.solve(timepoints, outputquantities)




fig = plt.figure(figsize=(15,5))
time = np.linspace(0, 3, 2000)
plt.plot([t * 1e9 for t in timepoints], output["mx1"], 'o', label=r"mx1")
plt.plot(time, ana(time), 'k--', label="Analytical")
plt.xlabel(r"time $t$ (ns)")
plt.ylabel(r"$M_x$")
plt.legend()



#popt, pcov = curve_fit(cosine, [t*1e9 for t in timepoints], output["mx"], p0=1.7e10)

#print("Analytical Larmor frequency: {} Hz".format(0.1 * 1.7595e11))
#print("Simulated frequency: {} Hz".format(popt[0]))

plt.savefig("cLLGtorque-test")
