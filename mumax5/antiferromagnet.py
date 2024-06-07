"""Antiferromagnet implementation."""

import numpy as _np

import _mumax5cpp as _cpp

from .fieldquantity import FieldQuantity
from .ferromagnet import Ferromagnet
from .parameter import Parameter


class Antiferromagnet:
    """Create an antiferromagnet instance.

    Parameters
    ----------
    world : mumax5.World
        World in which the ferromagnet lives.
    grid : mumax5.Grid
        The number of cells in x, y, z the ferromagnet should be divided into.
    geometry : None, ndarray, or callable (default=None)
        The geometry of the ferromagnet can be set in three ways.
        1. If the geometry contains all cells in the grid, then use None (the default)
        2. Use an ndarray which specifies for each cell wheter or not it is in the
           geometry.
        3. Use a function which takes x, y, and z coordinates as arguments and returns
           true if this position is inside the geometry and false otherwise.
    name : str (default="")
        The ferromagnet's identifier. If the name is empty (the default), a name for the
        ferromagnet will be created.
    """

    def __init__(self, world, grid, name="", geometry=None):

        if geometry is None:
            self._impl = world._impl.add_antiferromagnet(grid._impl, name)
            return

        if callable(geometry):
            # construct meshgrid of x, y, and z coordinates for the grid
            nx, ny, nz = grid.size
            cs = world.cellsize
            idxs = _np.flip(_np.mgrid[0:nz, 0:ny, 0:nx], axis=0)  # meshgrid of indices
            x, y, z = [(grid.origin[i] + idxs[i]) * cs[i] for i in [0, 1, 2]]

            # evaluate the geometry function for each position in this meshgrid
            geometry_array = _np.vectorize(geometry, otypes=[bool])(x, y, z)

        else:
            # When here, the geometry is not None, not callable, so it should be an
            # ndarray or at least should be convertable to ndarray
            geometry_array = _np.array(geometry, dtype=bool)
            if geometry_array.shape != grid.shape:
                raise ValueError(
                    "The dimensions of the geometry do not match the dimensions "
                    + "of the grid."
                )

        self._impl = world._impl.add_antiferromagnet(grid._impl, geometry_array, name)

    def __repr__(self):
        """Return Antiferromagnet string representation."""
        return f"Antiferromagnet(grid={self.grid}, name='{self.name}')"

    @classmethod
    def _from_impl(cls, impl):
        antiferromagnet = cls.__new__(cls)
        antiferromagnet._impl = impl
        return antiferromagnet

    @property
    def name(self):
        """Name of the antiferromagnet."""
        return self._impl.name

    @property
    def sub1(self):
        """First sublattice instance."""
        return Ferromagnet(self._impl.sub1)
    
    @property
    def sub2(self):
        """Second sublattice instance."""
        return Ferromagnet(self._impl.sub2)
    
    # ----- MATERIAL PARAMETERS -----------

    @property
    def afmex_cell(self):
        """Intercell antiferromagnetic exchange constant."""
        return Parameter(self._impl.afmex_cell)

    @afmex_cell.setter
    def afmex_cell(self, value):
        assert value <= 0, "The antiferromagnetic exchange constant afmex_cell should be negative (or zero)."
        self.afmex_cell.set(value)

    @property
    def afmex_nn(self):
        """Intracell antiferromagnetic exchange constant."""
        return Parameter(self._impl.afmex_nn)

    @afmex_nn.setter
    def afmex_nn(self, value):
        assert value <= 0, "The antiferromagnetic exchange constant afmex_nn should be negative (or zero)."
        self.afmex_nn.set(value)

    @property
    def latcon(self):
        """Lattice constant.
        Default = 0.35 nm.
        """
        return Parameter(self._impl.latcon)
    
    @latcon.setter
    def latcon(self, value):
        self.latcon.set(value)

    # ----- QUANTITIES ----------------------

    @property
    def neel_vector(self):
        """Neel vector of an antiferromagnet instance."""
        return FieldQuantity(_cpp.neel_vector(self._impl))