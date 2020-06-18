#!/bin/env python3

# This script solves micromagnetic standard problem 4. The Problem specification
# can be found on https://www.ctcms.nist.gov/~rdm/mumag.org.html

from mumax5 import World, Grid, TimeSolver, Table
from mumax5.util import show_field

import matplotlib.pyplot as plt
import numpy as np

length, width, thickness = 500e-9, 125e-9, 3e-9
nx, ny, nz = 128, 32, 1

world = World(cellsize=(length/nx, width/ny, thickness/nz))

magnet = world.add_ferromagnet(Grid((nx, ny, nz)))
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.02
magnet.temperature = 200

magnet.magnetization = (1, 0.1, 0)

solver = TimeSolver(magnet.magnetization, magnet.torque, magnet.thermal_noise)
solver.run(1e-10)

show_field(magnet.magnetization)