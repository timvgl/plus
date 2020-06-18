from mumax5 import World, Grid
import numpy as np


class TestFerromagnet:

    def test_magnetization_normalization(self):
        nx, ny, nz = 4,7,3

        w = World(cellsize=(1e-9, 1e-9, 1e-9))
        magnet = w.add_ferromagnet(Grid((nx, ny, nz)))

        m_not_normalized = 10*np.random.rand(3,nz,ny,nx)-5
        magnet.magnetization.set(m_not_normalized)

        m = magnet.magnetization.get()
        norms = np.linalg.norm(m,axis=0)

        assert np.max(np.abs(norms-1)) < 1e-5
