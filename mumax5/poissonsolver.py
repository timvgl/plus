class PoissonSolver:

    def __init__(self, impl):
        self._impl = impl

    def solve(self):
        """ Solve the Poisson equation """
        return self._impl.solve()

    def set_method(self, method_name):
        """ Set the solver method

        The implemented methods are:
            - jacobi
            - conjugategradient
        """
        return self._impl.set_method(method_name)

    @property
    def max_iter(self):
        """ Maximum of iterations for the solver (no maximum if <0)"""
        return self._impl.max_iter

    @max_iter.setter
    def max_iter(self, max_iter):
        self._impl.max_iter = max_iter

    @property
    def tol(self):
        """ Error tollerance on the max norm of the residual """
        return self._impl.tol

    @tol.setter
    def tol(self, tol):
        self._impl.tol = tol

    def max_norm_residual(self):
        """ Returns the maximum norm of the residual """
        return self._impl.max_norm_residual()

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
