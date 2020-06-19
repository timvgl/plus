import _mumax5cpp as _cpp


class World:
    """ World
     """

    def __init__(self, cellsize):
        """ Construct a world with a given cell size """
        self._impl = _cpp.World(cellsize)

    def add_ferromagnet(self, grid, name=""):
        """ Add a ferromagnet to this world

        Parameters
        ----------
        grid : Grid
            The grid on which the magnet lives
        name : string, optional
            The name of the magnet. If the string is empty, a name will be
            given by the engine.
        """

        return self._impl.add_ferromagnet(grid, name)

    def get_ferromagnet(self, name):
        """ Get a ferromagnet by its name """

        return self._impl.get_ferromagnet(name)

    @property
    def cellsize(self):
        """ The cell size of the world

        This property is a tuple of three floating pointing numbers which
        represent the dimensions of the cells in the x, y, and z direction.
        """
        return self._impl.cellsize

    @property
    def bias_magnetic_field(self):
        """ A uniform magnetic field which extends over the whole world """
        return self._impl.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self._impl.bias_magnetic_field = value
