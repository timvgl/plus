import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import *

import math

def helical(x, y, z):
    kx = k
    mx = math.cos(kx*x)
    my = math.sin(kx*x)
    mz = 0   
    return mx, my, mz

global k
k = 0

Ms = 800e3
mu0 = 4 * np.pi * 1e-7


length, width, thickness = 100e-6, 1e-9, 1e-9
V = length * width * thickness
Nx = int(length * 1e9)
nx, ny, nz = Nx, 1, 1
world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)), 3)
magnet.msat = Ms
#magnet.msat2 = 800e3
magnet.aex = 5e-12
#magnet.aex2 = 5e-12

magnet.magnetization = helical

angles, E, ana = [], [], []

fac = (0.5 *  mu0 * Ms * Ms) * V
kk = 0
while magnet.max_angle.eval() < np.pi-.01:
    k = kk * 1e7
    magnet.magnetization = helical

    E.append(magnet.exchange_energy.eval() / fac)
    angles.append(magnet.max_angle.eval() * 180 / np.pi)
    #angles.append(kk * 180/np.pi)
    ana.append(5e-12 * k**2 / fac * V)


    kk += 1

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8,6), gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0)
ax1.plot(angles, E, 'o', label=r"Mumax$^5$: FM")
ax1.plot(angles, ana, 'k--', label="Analytical")
ax1.set_xlim(0,180)
plt.xlabel(r"spin-spin angle (deg)")
ax1.set_ylabel(r"Energy density $\varepsilon / K_m V$")
ax1.legend()
error = [np.abs(a-e)/a for (e,a) in zip(E[1:], ana[1:])]
ax2.plot(angles[1:], error)
ax2.set_yscale("log")
ax2.set_ylabel("Relative error")
ax2.set_ylim(0.0001, 1)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))



plt.savefig("exchange-test")
