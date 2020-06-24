import numpy as np
from mumax5 import World, Grid, Ferromagnet

X, Y, Z = 0, 1, 2


def compute_interfacialdmi_numpy(magnet):
    m = magnet.magnetization.get()
    cs = magnet.cellsize
    h = np.zeros(m.shape)

    # -x neighbor
    m_ = np.roll(m, 1, axis=3)
    h[X][:, :, 1:] -= m_[Z][:, :, 1:] / cs[0]
    h[Z][:, :, 1:] += m_[X][:, :, 1:] / cs[0]

    # +x neighbor
    m_ = np.roll(m, -1, axis=3)
    h[X][:, :, 0:-1] += m_[Z][:, :, 0:-1] / cs[0]
    h[Z][:, :, 0:-1] -= m_[X][:, :, 0:-1] / cs[0]

    # -y neighbor
    m_ = np.roll(m, 1, axis=2)
    h[Y][:, 1:, :] -= m_[Z][:, 1:, :] / cs[1]
    h[Z][:, 1:, :] += m_[Y][:, 1:, :] / cs[1]

    # +y neighbor
    m_ = np.roll(m, -1, axis=2)
    h[Y][:, 0:-1, :] += m_[Z][:, 0:-1, :] / cs[1]
    h[Z][:, 0:-1, :] -= m_[Y][:, 0:-1, :] / cs[1]

    idmi = magnet.idmi.average()[0]
    msat = magnet.msat.average()[0]
    h *= idmi/msat
    return h


class TestInterfacialDmi:
    def test_interfacialdmi(self):
        cellsize = (1.1, 2.0, 3.2)
        world = World(cellsize)
        magnet = Ferromagnet(world, Grid((4, 5, 3)))
        magnet.msat = 3.1
        magnet.idmi = 7.1

        result = magnet.interfacialdmi_field.eval()
        wanted = compute_interfacialdmi_numpy(magnet)

        relative_error = np.abs(result-wanted)/np.abs(wanted)
        max_relative_error = np.max(relative_error)

        assert max_relative_error < 1e-4
