import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import *

"""
Switching of an MRAM bit.
--- Test of Slonczewski STT ---
"""

def ellips(a, b):
    a /= 2
    b /= 2
    
    def func(x, y, z):
        x -= 80e-9
        y -= 40e-9

        return x*x / (a*a) + y*y / (b*b) <= 1
              
    return func

length, width, thickness = 160e-9, 80e-9, 5e-9
nx, ny, nz = 64, 32, 1
total = nx * ny * nz
world = World(cellsize=(length / nx, width / ny, thickness / nz))


geo = lambda x, y, z: (x - 8E-8) ** 2 *4/ (length * length) +  (y -  4E-8) ** 2 *4/ (width * width) <= 1


magnet = Ferromagnet(world, Grid((nx, ny, nz)), 3, "magnet", geometry=geo)
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.01
magnet.magnetization = (1, 0, 0)
print(magnet.geometry)

print(magnet.magnetization.eval())
print(magnet.magnetization.average())
unique, counts = np.unique(magnet.geometry, return_counts=True)

show_field(magnet.magnetization)
plt.savefig("ellips")

print(dict(zip(unique, counts)))

magnet.pol = 0.5669
magnet.Lambda = 1
magnet.eps_prime = 0
area = np.pi * length * width / 4
magnet.jcur = (0, 0, 0.006 / area)

theta = 20 * np.pi/180
magnet.FixedLayer = (-np.cos(theta), -np.sin(theta), 0)
#magnet.FreeLayerThickness = 5e-9

timepoints = np.linspace(0, 2e-9, 200)
outputquantities = {"mx": lambda: magnet.magnetization.average()[0],
                    "my": lambda: magnet.magnetization.average()[1],
                    "mz": lambda: magnet.magnetization.average()[2]}
output = world.timesolver.solve(timepoints, outputquantities)

fig = plt.figure()
for key in outputquantities.keys():
    plt.plot(timepoints, output[key], '-', label=key)
plt.legend()
plt.ylim(-1, 1)
plt.savefig("ellipsgraph")

show_field(magnet.magnetization)
plt.savefig("ellipspost")