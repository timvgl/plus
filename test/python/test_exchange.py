import numpy as np

from mumax5.engine import World, Grid


def compute_exchange_numpy(magnet):
    m = magnet.magnetization.get()
    cellsize = magnet.cellsize
    h_exch = np.zeros(m.shape)

    m_ = np.roll(m, 1, axis=1)
    h_exch[:, 1:, :, :] += (m_-m)[:, 1:, :, :] / (cellsize[2]**2)

    m_ = np.roll(m, -1, axis=1)
    h_exch[:, :-1, :, :] += (m_-m)[:, :-1, :, :] / (cellsize[2]**2)

    m_ = np.roll(m, 1, axis=2)
    h_exch[:, :, 1:, :] += (m_-m)[:, :, 1:, :] / (cellsize[1]**2)

    m_ = np.roll(m, -1, axis=2)
    h_exch[:, :, 0:-1, :] += (m_-m)[:, :, 0:-1, :] / (cellsize[1]**2)

    m_ = np.roll(m, 1, axis=3)
    h_exch[:, :, :, 1:] += (m_-m)[:, :, :, 1:] / (cellsize[0]**2)

    m_ = np.roll(m, -1, axis=3)
    h_exch[:, :, :, 0:-1] += (m_-m)[:, :, :, 0:-1] / (cellsize[0]**2)

    return 2*magnet.aex.average()[0]*h_exch/magnet.msat.average()[0]


class TestExchange:
    def test_exchange(self):

        world = World((1e3, 2e3, 3e3))
        magnet = world.add_ferromagnet(Grid((16, 16, 4)))
        magnet.aex = 3.2e7
        magnet.msat = 5.4

        result = magnet.exchange_field.eval()
        wanted = compute_exchange_numpy(magnet)

        relative_error = np.abs(result-wanted)/np.abs(wanted)
        max_relative_error = np.max(relative_error)

        assert max_relative_error < 1e-4
