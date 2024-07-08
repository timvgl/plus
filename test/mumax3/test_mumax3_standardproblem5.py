import pytest
import numpy as np

from mumax3 import Mumax3Simulation
from mumax5 import Ferromagnet, Grid, World

ATOL = 1e-3  # 0.1%

def max_absolute_error(result, wanted):
    return np.max(abs(result - wanted))

xi1, xi2, xi3, xi4 = 0.0, 0.05, 0.1, 0.5
@pytest.fixture(scope="class", params=[xi1, xi2, xi3, xi4])
def simulations(request):
    """Sets up and runs standard problem 5 for both mumax5 and mumax3, given
    a specific non-adiabacity xi. The magnetization throughout time can later
    be compared quickly.
    This is very slow, but it rigorously tests spin transfer torque.
    """

    # === specifications ===
    length, width, thickness = 100e-9, 100e-9, 10e-9
    nx, ny, nz = 50, 50, 5  # following mumax3 paper
    cellsize = (length/nx, width/ny, thickness/nz)
    gridsize = (nx, ny, nz)

    msat = 800e3
    aex = 13e-12
    alpha = 0.1

    xi = request.param
    pol = 1.0  # purely polarized current
    jcur = (1e12, 0, 0)

    max_time = 5e-9
    step_time = 5e-12

    # === mumax3 ===
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{tuple(cellsize)}
            setgridsize{tuple(gridsize)}

            msat = {msat}
            aex = {aex}
            alpha = {alpha}

            m = vortex(1, 1)
            minimize()
            SaveAs(m, "m_initial.ovf")

            xi = {xi}
            Pol = {pol}
            J = vector{tuple(jcur)}

            tableautosave({step_time})
            run({max_time})
        """
    )

    # === mumax5 ===

    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))

    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha

    magnet.magnetization = mumax3sim.get_field("m_initial")
    magnet.minimize()

    magnet.xi = xi
    magnet.pol = pol
    magnet.jcur = jcur

    timepoints = np.arange(0, max_time + 0.5*step_time, step_time)
    outputquantities = {
        "t": lambda: world.timesolver.time,
        "mx": lambda: magnet.magnetization.average()[0],
        "my": lambda: magnet.magnetization.average()[1],
        "mz": lambda: magnet.magnetization.average()[2],
    }
    mumax5output = world.timesolver.solve(timepoints, outputquantities)

    return mumax5output, mumax3sim

@pytest.mark.slow
@pytest.mark.mumax3
class TestStandardProblem5:
    """Compare the results of standard problem #5 of mumax5 against mumax3.
    Standard Problems: http://www.ctcms.nist.gov/~rdm/mumag.org.html
    Number 5: https://www.ctcms.nist.gov/~rdm/std5/spec5.xhtml
    """

    def test_magnetization_x(self, simulations):
        mumax5output, mumax3sim = simulations
        # absolute error: mx goes through 0, but is unitless
        err = max_absolute_error(
            result=mumax5output["mx"], wanted=mumax3sim.get_column("mx")
        )
        print(err)
        assert err < ATOL

    def test_magnetization_y(self, simulations):
        mumax5output, mumax3sim = simulations
        # absolute error: my goes through 0, but is unitless
        err = max_absolute_error(
            result=mumax5output["my"], wanted=mumax3sim.get_column("my")
        )
        print(err)
        assert err < ATOL

    def test_magnetization_z(self, simulations):
        mumax5output, mumax3sim = simulations
        # absolute error: mz goes through 0, but is unitless
        err = max_absolute_error(
            result=mumax5output["mz"], wanted=mumax3sim.get_column("mz")
        )
        print(err)
        assert err < ATOL
