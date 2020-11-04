from mumax5 import World, Grid, Ferromagnet
import numpy as np


class TestAverage:
    def test_average_magnetization(self):
        world = World((1, 1, 1))
        magnet = Ferromagnet(world, Grid((32, 32, 5), (-10, 3, 0)))
        m = magnet.magnetization.get()
        wanted = np.average(m, axis=(1, 2, 3))
        result = magnet.magnetization.average()
        for i in range(3):
            assert np.abs((wanted[i]-result[i])/result[i]) < 1e-5
