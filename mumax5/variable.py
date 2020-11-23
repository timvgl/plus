"""Varaible implementation."""

from .fieldquantity import FieldQuantity


class Variable(FieldQuantity):
    """Create a new variable."""

    def __init__(self, impl):
        super().__init__(impl)

    def set(self, value):
        """Set the variable value."""
        self._impl.set(value)

    def get(self):
        """Get the variable value."""
        return self._impl.get()
