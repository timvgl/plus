import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World

def polar(uni, cub):
    zeros = (0, 0, 0)
    if uni:
        magnet.kc1, magnet.kc2, magnet.kc3 = zeros
        magnet.kc12, magnet.kc22, magnet.kc32 = zeros
        magnet.anisC1 = zeros
        magnet.anisC2 = zeros

        (magnet.ku1, magnet.ku12) = uni[0], uni[0]
        (magnet.ku2, magnet.ku22) = uni[1], uni[1]
        magnet.anisU = (1, 0, 0)
    
    if cub:
        magnet.ku1, magnet.ku12 = 0, 0
        magnet.ku2, magnet.ku22 = 0, 0
        magnet.anisU = zeros

        (magnet.kc1, magnet.kc12) = cub[0], cub[0]
        (magnet.kc2, magnet.kc22) = cub[1], cub[1]
        (magnet.kc3, magnet.kc32) = cub[2], cub[2]
        magnet.anisC1 = (1, 0, 0)
        magnet.anisC2 = (0, 1, 0)
        
    angles = np.linspace(0, 2 * np.pi, 50)
    energies = []
    energies_theo = []

    for th in angles:

        magnet.magnetization = (np.cos(th), np.sin(th), 0, -np.cos(th), -np.sin(th), 0)
        energies.append(magnet.anisotropy_energy_density.eval()[0][0][0][0])

        mx = magnet.magnetization.average()[3]
        my = magnet.magnetization.average()[4]

        if uni:
            E_theo = (uni[0] * (-mx ** 2) - uni[1] * (mx ** 4))
        if cub:
            E_theo = (cub[0] * (mx ** 2) * (my ** 2) + cub[2] * (mx ** 4) * (my ** 4))
        energies_theo.append(E_theo)

    if uni:
        lab = "Ku_1" if uni[0] else "Ku_2"
        s = '-' if uni[0]<0 or uni[1]<0 else ''
        rticks = [-1000, 0, 1000]

    elif cub:
        lab = "Kc_1" if cub[0] else "Kc_3"
        s = '-' if cub[0]<0 or cub[2]<0 else ''
        rticks = [-250, 0, 250]

    plt.polar(angles, energies_theo, 'k--')
    plt.polar(angles, energies, 'o', label=r"${} = ${}1 kJ/m³".format(lab, s))
    ax.set_rticks(rticks)
    ax.set_rlabel_position(0)
    ax.set_xticklabels([])
    plt.legend()

length, width, thickness = 1e-9, 1e-9, 1e-9
nx, ny, nz = 1, 1, 1
world = World(cellsize=(length / nx, width / ny, thickness / nz))

magnet = Ferromagnet(world, Grid((nx, ny, nz)), 6)
magnet.msat = 800e3
magnet.msat2 = 800e3
magnet.aex = 5e-12
magnet.aex2 = 5e-12
magnet.afmex_nn, magnet.afmex_cell = -5e-12, -5e-12 # ---> No influence, already tested.
magnet.alpha = 0.01


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(221, projection="polar")
polar([1e3, 0], [])
polar([-1e3, 0], [])
ax = fig.add_subplot(222, projection="polar")
polar([0, 1e3], [])
polar([0, -1e3], [])

ax = fig.add_subplot(223, projection="polar")
polar([], [1e3, 0, 0])
polar([], [-1e3, 0, 0])
ax = fig.add_subplot(224, projection="polar")
polar([], [0, 0, 1e3])
polar([], [0, 0, -1e3])
plt.tight_layout()
plt.savefig("anisotropy-test")
