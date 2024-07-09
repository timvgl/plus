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

    @classmethod
    def _from_impl(cls, impl):
        world = cls.__new__(cls)
        world._impl = impl
        return world

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
    
    def relax(self):
        """Relax the state to an energy minimum.
        -----

        The system evolves in time without precession (pure damping) until
        the total energy (i.e. the sum of all magnets in this world) hits
        the noise floor.
        Hereafter, relaxation keeps on going until the maximum torque is
        minimized.

        See also RelaxTorqueThreshold property.
        """
        self._impl.relax()

    @property
    def RelaxTorqueThreshold(self):
        """Threshold torque used for relaxing the system (default = -1).
        If set to a negative value (default behaviour),
            the system relaxes until the total torque (i.e. the sum of all
            magnets in this world) is steady or increasing.
        If set to a positive value,
            the system relaxes until the total torque is smaller than or
            equal to this threshold.
        """
        return self._impl.RelaxTorqueThreshold
        
    @RelaxTorqueThreshold.setter
    def RelaxTorqueThreshold(self, value):
        assert value != 0, "The relax threshold should not be zero."
        self._impl.RelaxTorqueThreshold = value

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
