import pytest
from mumax5 import Grid


class TestGrid:

    def test_valid_args(self):

        valid_gridsizes = [(1, 1, 1), [10, 3, 1], [10, 3, 100], (1, 3, 100)]
        invalid_gridsizes = [(0, -1, 1), [10, 0, -1], (-1, 3, 100),
                             (4.1, 3.1, 1.2), (1, -1, 1), (2, 1), 5, (9, 2, 1, 1), 'a']

        valid_origins = [(-4, 3, 1), (0, 0, 0)]
        invalid_origins = [(-4, 3.2, 1), (1, 1), 0]

        # check if valid gridsizes and valid origins do not raise errors
        for gridsize in valid_gridsizes:
            for origin in valid_origins:
                Grid(gridsize, origin)
                Grid(gridsize)

        # check if invalid gridsize raises an error
        for gridsize in invalid_gridsizes:
            for origin in valid_origins:
                with pytest.raises((ValueError, TypeError)):
                    Grid(gridsize, origin)
                with pytest.raises((ValueError, TypeError)):
                    Grid(gridsize)

        # check if invalid origins raises an error
        for origin in invalid_origins:
            for gridsize in valid_gridsizes:
                with pytest.raises((ValueError, TypeError)):
                    Grid(gridsize, origin)

    def test_gridsize(self):
        gridsize = (12, 14, 18)
        g = Grid(gridsize)
        assert gridsize == tuple(g.size)

    def test_origin(self):
        gridsize = (12, 14, 18)
        origin = (-3, 2, 7)
        grid_default_origin = Grid(gridsize)
        grid_nondefault_origin = Grid(gridsize, origin)
        assert tuple(grid_default_origin.origin) == (0, 0, 0)
        assert tuple(grid_nondefault_origin.origin) == (-3, 2, 7)
