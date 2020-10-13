from mumax5 import *
from mumax5.util import *
import numpy as np

world = World(cellsize=(1,1,1))
magnet = Ferromagnet(world, Grid((64,64,1)))

p = np.zeros( magnet.applied_potential.eval().shape )
p[:] = np.nan
p[0,0, 0,20:40] =  1
p[0,0,-1,:] = -1

magnet.applied_potential = p

show_layer(magnet.electrical_potential)