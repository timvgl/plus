from mumax5.engine import World, Grid, TimeSolver

import numpy as np


class TestTimeSolver:

    def test_timesolver(self):
        w = World(cellsize=(1, 1, 1))

        magnet = w.addFerromagnet("magnet", grid=Grid((2, 2, 1)))
        magnet.alpha = 0.1
        magnet.ku1 = 2.2
        magnet.anisU = (0.5, 1, 0)
        magnet.aex = 1.2
        magnet.msat = 1.2e6

        m = magnet.magnetization.get()
        m = 2*np.random.rand(*m.shape)-1
        magnet.magnetization.set(m)

        dt = 0.01
        solver = TimeSolver(magnet.magnetization, magnet.torque, dt)

        m = magnet.magnetization.get()
        torque = magnet.torque.eval()
        m_py = m + dt*torque

        solver.step()

        m_mumax5 = magnet.magnetization.get()

        # normalize m_py
        magnet.magnetization.set(m_py)
        m_py = magnet.magnetization.get()

        relerr = (m_py-m_mumax5)/m_py
        TOL = 1e-5
        assert np.max(relerr) < TOL
