"""Antiferromagnet implementation."""

import numpy as _np
import warnings

import _mumax5cpp as _cpp

from .fieldquantity import FieldQuantity
from .ferromagnet import Ferromagnet
from .grid import Grid
from .parameter import Parameter


class Antiferromagnet:
    """Create an antiferromagnet instance.

    This class can also be used to create a Ferrimagnet instance since both sublattices
    are independently modifiable.

    Parameters
    ----------
    world : mumax5.World
        World in which the antiferromagnet lives.
    grid : mumax5.Grid
        The number of cells in x, y, z the antiferromagnet should be divided into.
    geometry : None, ndarray, or callable (default=None)
        The geometry of the antiferromagnet can be set in three ways.
        1. If the geometry contains all cells in the grid, then use None (the default)
        2. Use an ndarray which specifies for each cell wheter or not it is in the
           geometry.
        3. Use a function which takes x, y, and z coordinates as arguments and returns
           true if this position is inside the geometry and false otherwise.
    name : str (default="")
        The antiferromagnet's identifier. If the name is empty (the default), a name for the
        antiferromagnet will be created.
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

        self._impl = world._impl.add_antiferromagnet(grid._impl, name)

    def __repr__(self):
        """Return Antiferromagnet string representation."""
        return f"Antiferromagnet(grid={self.grid}, name='{self.name}')"

    def __setattr__(self, name, value):
        """Set AFM or sublattice properties.
        
            If the AFM doesn't have the named attribute, then the corresponding
            attributes of both sublattices are set.
            e.g. to set the saturation magnetization of both sublattices to the
            same value, one could use:
                antiferromagnet.msat = 800e3
            which is equal to
                antiferromagnet.sub1.msat = 800e3
                antiferromagnet.sub2.msat = 800e3
        """
        if hasattr(Antiferromagnet, name) or name == "_impl":
            #TODO: this won't work anymore if there would come a Magnet parent class.
            super().__setattr__(name, value)
        elif hasattr(Ferromagnet, name):
            setattr(self.sub1, name, value)
            setattr(self.sub2, name, value)
        else:
            raise AttributeError(
                r'Both Antiferromagnet and Ferromagnet have no attribute "{}".'.format(name))

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
    def grid(self):
        """Return the underlying grid of the antiferromagnet."""
        return Grid._from_impl(self._impl.system.grid)

    @property
    def world(self):
        """Return the World of which the antiferromagnet is a part."""
        # same world as sublattice world; this uses less imports
        return self.sub1.world
    
    @property
    def sub1(self):
        """First sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub1())
    
    @property
    def sub2(self):
        """Second sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub2())
    
    @property
    def sublattices(self):
        return (self.sub1, self.sub2)

    @property
    def bias_magnetic_field(self):
        """Uniform bias magnetic field which will affect an antiferromagnet.

        The value should be specifed in Teslas.
        """
        return self.sub1.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self.sub1.bias_magnetic_field.set(value)
        self.sub2.bias_magnetic_field.set(value)

    def minimize(self):
        """Minimize the total energy.
        Fast energy minimization, but less robust than "relax"
        when starting from a high energy state.
        """
        self._impl.minimize()

    def relax(self, tol=1e-9):
        """Relax the state to an energy minimum.
        -----

        The system evolves in time without precession (pure damping) until
        the total energy (i.e. the sum of sublattices) hits the noise floor.
        Hereafter, relaxation keeps on going until the maximum torque is
        minimized.

        The tolerance argument corresponds to the maximum error of the timesolver.

        See also RelaxTorqueThreshold property of Ferromagnet.
        """
        if tol >= 1e-5:
            warnings.warn("The set tolerance is greater than or equal to the default value"
                          + " used for the timesolver (1e-5). Using this value results"
                          + " in no torque minimization, only energy minimization.", UserWarning)
        self._impl.relax(tol)


    # ----- MATERIAL PARAMETERS -----------

    @property
    def afmex_cell(self):
        """Intercell antiferromagnetic exchange constant."""
        return Parameter(self._impl.afmex_cell)

    @afmex_cell.setter
    def afmex_cell(self, value):
        if value > 0:
            warnings.warn("The antiferromagnetic exchange constant afmex_cell"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)
        self.afmex_cell.set(value)

    @property
    def afmex_nn(self):
        """Intracell antiferromagnetic exchange constant."""
        return Parameter(self._impl.afmex_nn)

    @afmex_nn.setter
    def afmex_nn(self, value):
        if value > 0:
            warnings.warn("The antiferromagnetic exchange constant afmex_nn"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)
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
        """Neel vector of an antiferromagnet instance.
        This quantity is defined as L = (M1 - M2) / 2
        """
        return FieldQuantity(_cpp.neel_vector(self._impl))
    
    @property
    def total_magnetization(self):
        """Total antiferromagnetic magnetization: M1 + M2."""
        return FieldQuantity(_cpp.total_magnetization(self._impl))