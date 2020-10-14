from .fieldquantity import FieldQuantity


class Variable(FieldQuantity):
    def __init__(self, impl):
        super().__init__(impl)

    def set(self, value):
        self._impl.set(value)

    def get(self):
        return self._impl.get()
