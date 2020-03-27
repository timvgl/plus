import pytest

from mumax5 import *

class TestFerromagnet:

    def test_anisotropy(self):
        w = World(cellsize=(1e-9,1e-9,1e-9))

        magnet = w.addFerromagnet("magnet", grid=Grid((2,2,1)))

        magnet.ku1 = 3
        magnet.anisU = (0,1,0)

        fanis = magnet.anisotropy_field.eval()
        print(fanis)