import _mumax5cpp as _cpp


class TimeSolverOutput:
    def __init__(self, quantity_dict):
        self._quantities = quantity_dict
        self._data = {"time": []}
        for key in self._quantities.keys():
            self._data[key] = []

    def write_line(self, time):
        self._data["time"].append(time)
        for key, func in self._quantities.items():
            self._data[key].append(func())

    def __getitem__(self, key):
        return self._data[key]


class TimeSolver:

    def __init__(self, variable, rhs):
        self._impl = _cpp.TimeSolver(variable, rhs._impl)

    def step(self):
        self._impl.step()

    def steps(self, nsteps):
        for i in range(nsteps):
            self.step()

    def run(self, duration):
        self._impl.run(duration)

    def solve(self, timepoints, quantity_dict):
        # check if time points are OK
        output = TimeSolverOutput(quantity_dict)
        for tp in timepoints:
            duration = tp - self.time
            self.run(duration)
            output.write_line(tp)
        return output

    @property
    def timestep(self):
        return self._impl.timestep

    @timestep.setter
    def timestep(self, timestep):
        self._impl.timestep = timestep

    @property
    def adaptive_timestep(self):
        return self._impl.adaptive_timestep

    @adaptive_timestep.setter
    def adaptive_timestep(self, adaptive):
        self._impl.timestep = adaptive

    @property
    def time(self):
        return self._impl.time
