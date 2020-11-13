class PoissonSolver:

    def __init__(self, impl):
        self._impl = impl

    def solve(self):
        """ Solve the Poisson equation """
        return self._impl.solve()

    # ----------------------------------------------------------------------------------
    # The hidden methods below should only be used to test and debug the Poisson solver

    def _init(self):
        """ Initializes/resets the poisson solver """
        self._impl.init()

    def _step(self):
        """ Let the solver perform a single step """
        self._impl.step()

    def _state(self):
        """ Return the current state of the Poisson solver """
        return self._impl.state()
