from mumax3 import Mumax3Simulation
from mumax5.engine import World, Grid

import numpy as np


def max_relative_error(result, wanted):
    err = np.linalg.norm(result-wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)


class TestMumax3:
    """ Test the effective fields of mumax5 against mumax3 """

    def setup_class(self):
        """ 1. Creates a magnet with arbitrary material parameters and grid
            2. Creates a mumax3 simulation with the same parameters
            3. Copies the magnetization of the mumax3 simulation to the magnet

            The effective fields of the magnet should now match the effective fields
            of the mumax3 simulation.
        """

        self.world = World((1e-9, 2e-9, 3.2e-9))
        self.magnet = self.world.add_ferromagnet(
            "magnet", Grid((29, 16, 4), (6, -3, 0)))
        self.magnet.msat = 800e3
        self.magnet.aex = 13e-12
        self.magnet.ku1 = 4.1e6
        self.magnet.anisU = (-0.3, 0, 1.5)

        self.mumax3sim = Mumax3Simulation(f"""
                setcellsize{tuple(self.world.cellsize)}
                setgridsize{tuple(self.magnet.grid.size)}
                msat = {self.magnet.msat}
                aex = {self.magnet.aex}
                ku1 = {self.magnet.ku1}
                anisU = vector{tuple(self.magnet.anisU)}
                m = neelskyrmion(1,1)
                saveas(m,"m.ovf")
                saveas(b_exch,"b_exch.ovf")
                saveas(b_anis,"b_anis.ovf")
                saveas(b_demag,"b_demag.ovf")
                saveas(b_eff,"b_eff.ovf")
                saveas(lltorque,"lltorque.ovf")
            """)

        self.magnet.magnetization.set(self.mumax3sim.get_field("m"))

    def test_magnetization(self):
        err = max_relative_error(result=self.magnet.magnetization.get(),
                                 wanted=self.mumax3sim.get_field("m"))
        assert err < 1e-3

    def test_anisotropy_field(self):
        err = max_relative_error(result=self.magnet.anisotropy_field.eval(),
                                 wanted=self.mumax3sim.get_field("b_anis"))
        assert err < 1e-3

    def test_exchange_field(self):
        err = max_relative_error(result=self.magnet.exchange_field.eval(),
                                 wanted=self.mumax3sim.get_field("b_exch"))
        assert err < 1e-3

    def test_demag_field(self):
        # Here we compare to the demagfield of mumax with an increased tollerance.
        # Because mumax3 and mumax5 approximate in a different way the demag kernel
        err = max_relative_error(result=self.magnet.demag_field.eval(),
                                 wanted=self.mumax3sim.get_field("b_demag"))
        assert err < 1e-2

    def test_effective_field(self):
        # Here we compare to the demagfield of mumax with an increased tollerance.
        # Because mumax3 and mumax5 approximate in a different way the demag kernel
        err = max_relative_error(result=self.magnet.effective_field.eval(),
                                 wanted=self.mumax3sim.get_field("b_eff"))
        assert err < 1e-2
