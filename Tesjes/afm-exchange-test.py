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
    return mx, my, mz, -mx, -my, -mz

def analytical(A, A0, A12, k):
    fm = A * k * k
    intra = -4 * A0 * (-1) / (0.35e-9 * 0.35e-9)
    inter = A12 * (-k * k)
    return (fm + fm + intra + inter) / 2

global k
k = 0

Ms = 800e3
A = 5e-12
A0 = -10e-12
A12 = -5e-12
mu0 = 4 * np.pi * 1e-7


length, width, thickness = 100e-6, 1e-9, 1e-9
V = length * width * thickness
Nx = int(length * 1e9)
nx, ny, nz = Nx, 1, 1
world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)), 6)
magnet.msat = Ms
magnet.msat2 = Ms
magnet.aex = A
magnet.aex2 = A
magnet.afmex_cell = A0
magnet.afmex_nn = A12


magnet.magnetization = helical

angles, E, E2, ana = [], [], [], []

fac = (0.5 *  mu0 * Ms * Ms) * V
kk = 0
while magnet.max_angle.eval() < np.pi-.01:
    k = kk * 1e7
    magnet.magnetization = helical

    E.append(magnet.exchange_energy.eval() / fac)
    E2.append(magnet.exchange_energy2.eval() / fac)
    angles.append(magnet.max_angle.eval() * 180 / np.pi)
    
    ana.append(analytical(A, A0, A12, k) / fac * V)


    kk += 1

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8,6), gridspec_kw={'height_ratios': [2, 1]})
fig.subplots_adjust(hspace=0)
ax1.plot(angles, E, 'o', label=r"Mumax$^5$: Sublattice 1")
ax1.plot(angles, E2, '-', label=r"Mumax$^5$: Sublattice 2")
ax1.plot(angles, ana, 'k--', label="Analytical")
ax1.set_xlim(0,180)
plt.xlabel(r"spin-spin angle (deg)")
ax1.set_ylabel(r"Energy density $\varepsilon / K_m V$")
ax1.legend()
error = [np.abs((a-e)/a) for (e,a) in zip(E[1:], ana[1:])]
ax2.plot(angles[1:], error)
ax2.set_yscale("log")
ax2.set_ylabel("Relative error")
ax2.set_ylim(1e-6, 1)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))



plt.savefig("afm-exchange-test")
