import pytest

from mumax3 import Mumax3Simulation

class TestMumax3:

    def setup_class(self):
        self.mumax3sim = Mumax3Simulation("""
                setcellsize(1,1,1)
                setgridsize(16,16,1)
                save(m)
            """)

    def test_sim(self):
        m = self.mumax3sim.get_field("m000000")
        assert m.shape == (3,1,16,16)