import pytest

from mumax3 import Mumax3Simulation
from mumax5 import *

import numpy as np


def fields_equal(result, wanted, reltol=1e-5):
    err = np.linalg.norm(result-wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    maxrelerr = np.max(relerr)
    return maxrelerr < reltol


class TestMumax3:
    """ Test the effective fields of mumax5 against mumax3 """

    def setup_class(self):
        """ 1. Creates a magnet with arbitrary material parameters and grid
            2. Creates a mumax3 simulation with the same parameters
            3. Copies the magnetization of the mumax3 simulation to the magnet

            The effective fields of the magnet should now match the effective fields
            of the mumax3 simulation.
        """

        self.world = World((1e-9, 2e-9, 1.5e-9))
        self.magnet = self.world.addFerromagnet("magnet", Grid((6, 4, 5)))
        self.magnet.msat = 3.2e5
        self.magnet.aex = 3.4
        self.magnet.ku1 = 7.1e6
        self.magnet.anisU = (-0.3, 0, 1.5)

        self.mumax3sim = Mumax3Simulation(f"""
                setcellsize{tuple(self.world.cellsize())}
                setgridsize{tuple(self.magnet.grid().size)}
                msat = {self.magnet.msat}
                aex = {self.magnet.aex}
                ku1 = {self.magnet.ku1}
                anisU = vector{tuple(self.magnet.anisU)}
                saveas(m,"m.ovf")
                saveas(b_exch,"b_exch.ovf")
                saveas(b_anis,"b_anis.ovf")
                saveas(b_demag,"b_demag.ovf")
            """)

        self.magnet.magnetization.set(self.mumax3sim.get_field("m"))

    def test_magnetization(self):
        assert fields_equal(result=self.magnet.magnetization.get(),
                            wanted=self.mumax3sim.get_field("m"))

    def test_anisotropy_field(self):
        assert fields_equal(result=self.magnet.anisotropy_field.eval(),
                            wanted=self.mumax3sim.get_field("b_anis"))

    def test_exchange_field(self):
        assert fields_equal(result=self.magnet.exchange_field.eval(),
                            wanted=self.mumax3sim.get_field("b_exch"))

    def test_demag_field(self):
        assert fields_equal(result=self.magnet.demag_field.eval(),
                            wanted=self.mumax3sim.get_field("b_demag"))
