import _mumax5cpp as _cpp


class Grid:
    def __init__(self, size, origin=(0, 0, 0)):
        self._impl = _cpp.Grid(size, origin)

    @classmethod
    def _from_impl(cls, impl):
        grid = cls.__new__(cls)
        grid._impl = impl
        return grid

    @property
    def size(self):
        return self._impl.size

    @property
    def origin(self):
        return self._impl.origin

    @property
    def shape(self):
        Nx, Ny, Nz = self.size
        return (Nz, Ny, Nx)

    @property
    def ncells(self):
        Nx, Ny, Nz = self.size
        return Nx*Ny*Nz
