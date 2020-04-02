import pytest

from mumax5.engine import World


class TestWorld:

    def test_init_args(self):
        valid_cellsizes = [(1.0, 3.2, 5.1), (3.2, 3.2, 3.2), [1e-8, 1e-9, 0.4]]
        invalid_cellsizes = [(-1.0, 3.2, 5.1), (3.2, 3.2), 5.3]

        for cellsize in valid_cellsizes:
            World(cellsize)

        for cellsize in invalid_cellsizes:
            with pytest.raises((ValueError, TypeError)):
                World(cellsize)

    def test_cellsize(self):
        cellsize = (9.1, 1e-3, 3.2)
        w = World(cellsize)
        assert tuple(w.cellsize()) == cellsize
