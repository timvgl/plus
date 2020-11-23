"""Scalar quantity implementation."""


class ScalarQuantity:
    """Create a new scalar quantity."""

    def __init__(self, impl):
        self._impl = impl

    @property
    def name(self):
        """Return the instance's name."""
        return self._impl.name

    @property
    def unit(self):
        """Return the instance's unit."""
        return self._impl.unit

    def eval(self):
        """Evaluate and return scalar quantity."""
        return self._impl.eval()

    def __call__(self):
        """Evaluate and return scalar quantity."""
        return self.eval()
