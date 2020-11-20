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

magnet.conductivity = 1.0
cond = magnet.conductivity.eval()
cond[:, :, 50:60, 0:120] = 0.0
magnet.conductivity = cond

magnet.poisson_system._init()
solver = magnet.poisson_system._solver
solver.set_method("conjugategradient")

residual = []
for i in range(2000):
    solver.step()
    residual.append(solver.max_norm_residual())

pot = solver.state()

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.loglog(residual)

ax2.imshow(pot[0, 0])

plt.show()
