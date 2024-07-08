import numpy as np
import pytest
from mumax3 import Mumax3Simulation
from mumax5 import Ferromagnet, Grid, World

RTOL = 1e-3

@pytest.fixture(params=[True, False])
def openbc(request):
    return request.param

class TestBulkDMI:
    """Test bulk dmi against mumax3."""

    @pytest.fixture(autouse=True)
    def setup_class(self, openbc):
        # arbitrarily chosen parameters
        msat, dbulk = 800e3, 3e-3
        cellsize = (1e-9, 2e-9, 3.2e-9)
        gridsize = (30, 16, 4)

        self.mumax3sim = Mumax3Simulation(
            f"""
                setcellsize{cellsize}
                setgridsize{gridsize}
                msat = {msat}
                Dbulk = {dbulk}

                m = randommag()
                saveas(m, "m.ovf")
                
                // default in mumax3 is false, these tests use both
                openbc = {openbc}

                // The dmi is included in the exchange in mumax3
                // because Aex is set to zero here, b_exch is the dmi field
                saveas(b_exch, "b_dmi.ovf")      
                saveas(edens_exch, "edens_dmi.ovf") 
                saveas(e_exch, "e_dmi.ovf")
            """
        )

        self.world = World(cellsize)
        self.magnet = Ferromagnet(self.world, Grid(gridsize))
        self.magnet.enable_demag = False
        self.magnet.enable_openbc = openbc
        self.magnet.msat = msat
        self.magnet.dmi_tensor.set_bulk_dmi(dbulk)
        self.magnet.magnetization.set(self.mumax3sim.get_field("m"))

    def test_dmi_field(self, openbc):
        if not openbc: pytest.xfail("Known failure for closed BC")
        wanted = self.mumax3sim.get_field("b_dmi")
        result = self.magnet.dmi_field()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_in_effective_field(self, openbc):
        if not openbc: pytest.xfail("Known failure for closed BC")
        wanted = self.magnet.dmi_field()
        result = self.magnet.effective_field()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_energy_density(self, openbc):
        if not openbc: pytest.xfail("Known failure for closed BC")
        wanted = self.mumax3sim.get_field("edens_dmi")
        result = self.magnet.dmi_energy_density()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_in_total_energy_density(self, openbc):
        if not openbc: pytest.xfail("Known failure for closed BC")
        wanted = self.magnet.dmi_energy_density()
        result = self.magnet.total_energy_density()
        assert np.allclose(result, wanted, rtol=RTOL)

    def test_dmi_energy(self, openbc):
        if not openbc: pytest.xfail("Known failure for closed BC")
        wanted = self.mumax3sim.get_field("e_dmi").flat[0]  # same value in all cells
        result = self.magnet.dmi_energy()
        assert np.isclose(result, wanted, rtol=RTOL)

    def test_dmi_in_total_energy(self, openbc):
        if not openbc: pytest.xfail("Known failure for closed BC")
        wanted = self.magnet.dmi_energy()
        result = self.magnet.total_energy()
        assert np.isclose(result, wanted, rtol=RTOL)
