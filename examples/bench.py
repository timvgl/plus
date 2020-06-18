from mumax5 import *
import time

world = World((4e-9, 4e-9, 4e-9))

magnet = world.add_ferromagnet(Grid((128, 64, 1)))

magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.5

magnet.magnetization = (1, 0.1, 0)

solver = TimeSolver(magnet.magnetization, magnet.torque)
solver.timestep = 1e-13
solver.adaptive_timestep = False

nsteps = 1000
start = time.time()
solver.steps(nsteps)
print("#steps/s: ", round(nsteps/(time.time()-start)))