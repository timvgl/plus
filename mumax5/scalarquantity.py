class ScalarQuantity:
    def __init__(self, impl):
        self._impl = impl

    @property
    def name(self):
        return self._impl.name

    @property
    def unit(self):
        return self._impl.unit

    def eval(self):
        return self._impl.eval()

    def __call__(self):
        return self.eval()
