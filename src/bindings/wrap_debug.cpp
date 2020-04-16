#include "timer.hpp"
#include "wrappers.hpp"

void wrap_debug(py::module& m) {
  m.def("_show_timings", []() { timer.printTimings(); });
}