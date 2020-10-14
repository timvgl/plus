import _mumax5cpp as _cpp

from .fieldquantity import FieldQuantity


class MagnetField(FieldQuantity):
    def __init__(self, magnet, grid):
        super().__init__(_cpp.MagnetField(magnet._impl, grid._impl))

    def set_method(self, method):
        self._impl.set_method(method)
