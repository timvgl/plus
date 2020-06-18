from mumax5.engine import *
from mumax5.util import show_field

# infinite grid in z direction, periodic in x and y
mastergrid = Grid(size=(128, 128, 0), origin=(0, 0, 0))
cellsize = (0.1, 0.1, 0.1)

world = World(cellsize, mastergrid)

magnet = world.add_ferromagnet(grid=Grid((128, 128, 1)))
magnet.enable_demag = False
magnet.idmi = 1.5
magnet.msat = 1.0
magnet.aex = 1.0
magnet.ku1 = 1.0
magnet.anisU = (0, 0, 1)
magnet.alpha = 0.5

solver = TimeSolver(magnet.magnetization, magnet.torque)
solver.steps(500)

show_field(magnet.magnetization)
