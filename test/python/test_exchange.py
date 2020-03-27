import pytest
import numpy as np

from mumax5 import *

def getMeshGrid(grid, cellsize):
    o = grid.origin
    s = grid.size
    c = cellsize
    x = (np.linspace(0,s[0]-1,s[0])+o[0])*c[0]
    y = (np.linspace(0,s[1]-1,s[1])+o[1])*c[1]
    z = (np.linspace(0,s[2]-1,s[2])+o[2])*c[2]
    return np.meshgrid(z, y, x, indexing='ij')

class TestExchange:
    def test_exchange(self):
        w = World((1,1,1))
        fm = w.addFerromagnet("magnet", grid=Grid((4,3,2)))

        zz,yy,xx = getMeshGrid(fm.grid(),w.cellsize())

        mx = np.cos(xx)
        my = np.sin(xx)
        mz = 0*xx
        fm.magnetization.set( np.array([mx,my,mz]) )

        fm.aex = 10.0

        print(fm.magnetization.get())
        print(fm.exchange_field.eval())
        
