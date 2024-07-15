import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import *
from scipy.optimize import curve_fit

def line(x, a, b):
    return a*x+b

def lex(msat, aex):
    return np.sqrt(2*aex / (4*np.pi*1e-7*msat*msat))

length, width, thickness = 250e-9, 250e-9, 0.6e-9
nx, ny, nz = 128, 128, 1
world = World(cellsize=(length / nx, width / ny, thickness / nz))


magnet = Ferromagnet(world, Grid((nx, ny, nz)), 3)
magnet.msat = 1100e3
magnet.aex = 16e-12 # lex = 4.59 nm
magnet.alpha = 3
magnet.enable_openbc = False

magnet.ku1 = 1.27e6
magnet.anisU = (0, 0, 1)

#magnet.dmi_tensor.set_interfacial_dmi(0.1e-3)
magnet.magnetization = twodomain((0, 0, -1), (0, 0, 1), (1, 1, 0), length/2, 0.3e-9)


mx, my, mz, mtx, mty = [], [], [], [], []
energy = []
magnet.minimize()

Dsteps = 50
Dmax = 0.3
D = np.linspace(0, Dmax, Dsteps)

for d in D:

    magnet.dmi_tensor.set_interfacial_dmi(d * 1e-3)
    #magnet.dmi_tensor.set_bulk_dmi(d * 1e-3)

    magnet.minimize()
    mx.append(magnet.magnetization.average()[0])
    my.append(magnet.magnetization.average()[1])
    energy.append(magnet.dmi_energy.eval())

    #mtx.append(np.trapz(magnet.magnetization.eval()[0][0], axis=1).sum())
    #mty.append(np.trapz(magnet.magnetization.eval()[1][0], axis=1).sum())



fig = plt.figure(figsize=(7,5))
plt.plot(D, mx, 'o', label=r"$<m_x>$")
plt.plot(D, my, 'o', label=r"$<m_y>$")
plt.xlim(0, Dmax)
plt.ylim(0, 0.1)
plt.xlabel(r"$D_{int}$ (mJ/m²)")
plt.ylabel("Domain wall moments (a.u.)")
plt.legend()
plt.savefig("DMI_DW-test")


cutoff = Dsteps - 10
popt, pcov = curve_fit(line, D[cutoff:], energy[cutoff:])

fig = plt.figure()
plt.plot(D, energy, 'o')
plt.plot(D[cutoff:], line(D[cutoff:], *popt), 'k--')
plt.xlabel("D")
plt.ylabel("Energy")
plt.show()

