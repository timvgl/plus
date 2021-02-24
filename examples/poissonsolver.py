import matplotlib.pyplot as plt
import numpy as np

from mumax5 import *
from mumax5.util import *

length, width, height = 100e-9, 100e-9, 1e-9
nx, ny, nz = 128, 128, 1

world = World(cellsize=(length/nx, width/ny, height/nz))
magnet = Ferromagnet(world, Grid((nx, ny, nz)))

magnet.conductivity = 2.3


# Set the applied potential (nan is no applied potential)
def applied_potential(x,y,z):
    if y < 0.1*width:
        return 1
    elif y > 0.9*width and abs(x-length/2) < 0.4 * length:
        return -1
    else:
        return np.nan

magnet.applied_potential = applied_potential

magnet.poisson_system._init()
solver = magnet.poisson_system._solver
solver.set_method("conjugategradient")

residual = []
for i in range(2000):
    solver.step()
    residual.append(solver.max_norm_residual())

fig, (ax1) = plt.subplots(1, 1)
ax1.loglog(residual)

plt.show()
