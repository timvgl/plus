"""MagnetField implementation."""

import _mumax5cpp as _cpp

from .fieldquantity import FieldQuantity


class MagnetField(FieldQuantity):
    """Create a magnetic field instance.

    Parameters
    ----------
    magnet : mumax5.Ferromagnet
        Magnet instance which is the field source.
    grid : mumax5.Grid
        Grid instance, determines number of cells in the magnet.
    """

    def __init__(self, magnet, grid):
        super().__init__(_cpp.MagnetField(magnet._impl, grid._impl))

    def set_method(self, method):
        """Set the computation method for the stray field.

        Parameters
        ----------
        method : {"brute", "fft"}, optional
            The default value is "fft".
        """
        self._impl.set_method(method)
