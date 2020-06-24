#!/bin/env python3

# This script solves micromagnetic standard problem 4. The Problem specification
# can be found on https://www.ctcms.nist.gov/~rdm/mumag.org.html

from mumax5 import World, Grid, TimeSolver, Table, Ferromagnet

import matplotlib.pyplot as plt
import numpy as np

length, width, thickness = 500e-9, 125e-9, 3e-9
nx, ny, nz = 128, 32, 1

world = World(cellsize=(length/nx, width/ny, thickness/nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)))
magnet.msat = 800e3
magnet.aex = 13e-12
magnet.alpha = 0.02

magnet.magnetization = (1, 0.1, 0)
magnet.minimize()

world.bias_magnetic_field = (-24.6e-3, 4.3e-3, 0)
#world.bias_magnetic_field = (-35.5e-3, -6.3e-3, 0)

timepoints = np.linspace(0, 1e-9, 1000)

table = Table()
table.add("mx", magnet.magnetization, 0)
table.add("my", magnet.magnetization, 1)
table.add("mz", magnet.magnetization, 2)
#table.add("e_total", magnet.total_energy_density, 0)
#table.add("e_exchange", magnet.exchange_energy_density, 0)
#table.add("e_zeeman", magnet.zeeman_energy_density, 0)
#table.add("e_demag", magnet.demag_energy_density, 0)

solver = TimeSolver(magnet.magnetization, magnet.torque)
solver.solve(timepoints, table)

plt.subplot(211)
for key in ["mx", "my", "mz"]:
    plt.plot(timepoints, table[key], label=key)
plt.legend()

# plt.subplot(212)
# for key in ["e_total", "e_exchange", "e_zeeman", "e_demag"]:
#    plt.plot(timepoints, table[key], label=key)
# plt.legend()

plt.show()

#SetGridsize(128, 32, 1)
#SetCellsize(500e-9/128, 125e-9/32, 3e-9)
#
#Msat  = 800e3
#Aex   = 13e-12
#alpha = 0.02
#
#m = uniform(1, .1, 0)
# relax()
# save(m)    // relaxed state
#
#autosave(m, 200e-12)
# tableautosave(10e-12)
#
#B_ext = vector(-24.6E-3, 4.3E-3, 0)
# run(1e-9)
