import matplotlib.pyplot as plt
from mumax5 import *
from mumax5.util import *
import numpy as np

world = World(cellsize=(1, 1, 1))
magnet = Ferromagnet(world, Grid((256, 128, 1)))

p = np.zeros(magnet.applied_potential.eval().shape)
p[:] = np.nan
p[0, 0, 0, 20:40] = 1
p[0, 0, -1, :] = -1

magnet.applied_potential = p

magnet.poisson_solver.set_method("steepestdescent")
magnet.poisson_solver._init()

residual = []
for i in range(2000):
    magnet.poisson_solver._step()
    residual.append(magnet.poisson_solver.max_norm_residual())

pot = magnet.poisson_solver._state()

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.loglog(residual)

ax2.imshow(pot[0, 0])

plt.show()
