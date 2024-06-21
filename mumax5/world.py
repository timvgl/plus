"""World imlementation."""

import _mumax5cpp as _cpp

from .timesolver import TimeSolver
from .grid import Grid
from .ferromagnet import Ferromagnet
from .antiferromagnet import Antiferromagnet

class World:
    """Construct a world with a given cell size.

    Parameters
    ----------
    cellsize : tuple[float] of size 3
        A tuple of three floating pointing numbers which represent the dimensions
        of the cells in the x, y, and z direction.
    """

    def __init__(self, cellsize, mastergrid=Grid((0, 0, 0))):
        if len(cellsize) != 3:
            raise ValueError("'cellsize' should have three dimensions.")

        self._impl = _cpp.World(cellsize)

    def __repr__(self):
        """Return World string representation."""
        return f"World(cellsize={self.cellsize})"

    @property
    def timesolver(self):
        """Time solver for this world."""
        return TimeSolver(self._impl.timesolver)

    def get_ferromagnet(self, name):
        """Get a ferromagnet by its name.
        Raises KeyError if there is no magnet with the given name."""
        magnet_impl = self._impl.get_ferromagnet(name)
        if magnet_impl is None:
            raise KeyError(f"No magnet named {name}")
        return Ferromagnet._from_impl(magnet_impl)
    
    def get_antiferromagnet(self, name):
        """Get an antiferromagnet by its name.
        Raises KeyError if there is no magnet with the given name."""
        magnet_impl = self._impl.get_antiferromagnet(name)
        if magnet_impl is None:
            raise KeyError(f"No magnet named {name}")
        return Antiferromagnet._from_impl(magnet_impl)

    @property
    def ferromagnets(self):
        """Get a dictionairy of ferromagnets by name."""
        return {key: Ferromagnet._from_impl(impl) for key, impl in
                self._impl.ferromagnets.items()}
    
    @property
    def antiferromagnets(self):
        """Get a dictionairy of antiferromagnets by name."""
        return {key: Antiferromagnet._from_impl(impl) for key, impl in
                self._impl.antiferromagnets.items()}

    @property
    def cellsize(self):
        """Return the cell size of the world.

        Returns
        -------
        tuple[float] of size 3
            A tuple of three floating pointing numbers which represent the dimensions
            of the cells in the x, y, and z direction.
        """
        return self._impl.cellsize

    @property
    def bias_magnetic_field(self):
        """Return a uniform magnetic field which extends over the whole world."""
        return self._impl.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        """Set a uniform magnetic field which extends over the whole world."""
        self._impl.bias_magnetic_field = value
