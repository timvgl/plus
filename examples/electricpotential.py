import matplotlib.pyplot as plt
from mumax5 import *
from mumax5.util import *
import numpy as np

world = World(cellsize=(1, 1, 1))
magnet = Ferromagnet(world, Grid((64, 64, 1)))

p = np.zeros(magnet.applied_potential.eval().shape)
p[:] = np.nan
p[0, 0, 0, 20:40] = 1
p[0, 0, -1, :] = -1

magnet.applied_potential = p

# show_layer(magnet.electrical_potential)

magnet.poisson_solver._init()

for i in range(1000):
    magnet.poisson_solver._step()

pot = magnet.poisson_solver._state()
plt.imshow(pot[0, 0])
plt.show()
