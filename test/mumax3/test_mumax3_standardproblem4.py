from mumax3 import Mumax3Simulation
import numpy as np

from mumax5 import Ferromagnet, Grid, World

RTOL = 3e-2  # 3% is quite large :(

def max_absolute_error(result, wanted):
    return np.max(abs(result - wanted))

def max_relative_error(result, wanted):
    return np.max(abs((result - wanted)/wanted))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))

class TestStandardProblem4:
    """Compare the results of standard problem #4 of mumax5 against mumax3.
    Standard Problems: http://www.ctcms.nist.gov/~rdm/mumag.org.html
    Number 4: https://www.ctcms.nist.gov/~rdm/std4/spec4.html
    """

    def setup_class(self):
        """Sets up and runs standard problem 4 for both mumax5 and mumax3. The
        magnetization and energies throughout time can later be compared quickly.
        """

        # === specifications ===
        length, width, thickness = 500e-9, 125e-9, 3e-9
        nx, ny, nz = 128, 32, 1

        msat = 800e3
        aex = 13e-12
        alpha = 0.02

        magnetization = (1, 0.1, 0)

        B1 = (-24.6e-3, 4.3e-3, 0)  # a
        B2 = (-35.5e-3, -6.3e-3, 0)  # b
        # TODO: figure out a way to test both, without always running setup_class
        B_mag = B1  # choose B1 or B2 here

        max_time = 1e-9
        step_time = 1e-12
        
        # === mumax5 ===
        self.world = World(cellsize=(length / nx, width / ny, thickness / nz))
        self.magnet = Ferromagnet(self.world, Grid((nx, ny, nz)))
        
        self.magnet.msat = msat
        self.magnet.aex = aex
        self.magnet.alpha = alpha

        self.magnet.magnetization = magnetization
        self.magnet.minimize()

        self.world.bias_magnetic_field = B_mag

        timepoints = np.arange(0, max_time + 0.5*step_time, step_time)
        outputquantities = {
            "t": lambda: self.world.timesolver.time,
            "mx": lambda: self.magnet.magnetization.average()[0],
            "my": lambda: self.magnet.magnetization.average()[1],
            "mz": lambda: self.magnet.magnetization.average()[2],
            "E_total": self.magnet.total_energy,
            "E_exch": self.magnet.exchange_energy,
            "E_Zeeman": self.magnet.zeeman_energy,
            "E_demag": self.magnet.demag_energy
        }

        self.output = self.world.timesolver.solve(timepoints, outputquantities)


        # === mumax3 ===
        self.mumax3sim = Mumax3Simulation(
            f"""
                setcellsize{tuple(self.world.cellsize)}
                setgridsize{tuple(self.magnet.grid.size)}

                msat = {msat}
                aex = {aex}
                alpha = {alpha}

                m = uniform{tuple(magnetization)}
                minimize()

                tableadd(E_total)
                tableadd(E_exch)
                tableadd(E_zeeman)
                tableadd(E_demag)

                tableautosave({step_time})
                B_ext = vector{tuple(B_mag)}

                run({max_time})
            """
        )

    def test_magnetization_x(self):
        # absolute error: mx goes through 0, but is unitless
        err = max_absolute_error(
            result=self.output["mx"], wanted=self.mumax3sim.get_column("mx")
        )
        assert err < 2e-2

    def test_magnetization_y(self):
        # absolute error: my goes through 0, but is unitless
        err = max_absolute_error(
            result=self.output["my"], wanted=self.mumax3sim.get_column("my")
        )
        assert err < RTOL

    def test_magnetization_z(self):
        # absolute error: mz goes through 0, but is unitless
        err = max_absolute_error(
            result=self.output["mz"], wanted=self.mumax3sim.get_column("mz")
        )
        assert err < RTOL

    def test_total_energy(self):
        # semirelative: E_total goes through 0, but has a unit
        err = max_semirelative_error(
            result=self.output["E_total"], wanted=self.mumax3sim.get_column("E_total")
        )
        assert err < RTOL

    def test_exchange_energy(self):
        # relative: E_exch is always positive
        err = max_relative_error(
            result=self.output["E_exch"], wanted=self.mumax3sim.get_column("E_exch")
        )
        assert err < RTOL

    def test_zeeman_energy(self):
        # semirelative: E_Zeeman goes through 0, but has a unit
        err = max_semirelative_error(
            result=self.output["E_Zeeman"], wanted=self.mumax3sim.get_column("E_Zeeman")
        )
        assert err < RTOL

    def test_demag_energy(self):
        # relative: E_demag is probably always positive
        err = max_relative_error(
            result=self.output["E_demag"], wanted=self.mumax3sim.get_column("E_demag")
        )
        assert err < RTOL
