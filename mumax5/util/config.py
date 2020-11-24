"""Common magnetization configurations."""

import numpy as np


def set_magnetization(magnet, cellsize, config):
    """Set a magnetization configuration (e.g. the Neel skyrmion) on a magnet."""
    cs = cellsize
    gs = magnet.grid.size
    go = magnet.grid.origin
    m = np.zeros((3, gs[2], gs[1], gs[0]))
    for iz in range(gs[2]):
        for iy in range(gs[1]):
            for ix in range(gs[0]):
                x = (go[0] + ix) * cs[0]
                y = (go[1] + iy) * cs[1]
                z = (go[2] + iz) * cs[2]
                m[:, iz, iy, ix] = config(x, y, z)
    magnet.magnetization = m


def neelskyrmion(center, radius, charge, pol):
    """Create the Neel skyrmion magnetization configuration."""

    def func(x, y, z):
        x -= center[0]
        y -= center[1]
        r = np.sqrt(x ** 2 + y ** 2)
        if r == 0.0:
            return (0, 0, pol)
        mz = 2 * pol * (np.exp(-((r / radius) ** 2)) - 0.5)
        mx = (x * charge / r) * (1 - np.abs(mz))
        my = (y * charge / r) * (1 - np.abs(mz))
        return (mx, my, mz)

    return func
