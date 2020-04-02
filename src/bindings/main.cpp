#include "wrappers.hpp"

PYBIND11_MODULE(engine, m) {
  wrap_ferromagnet(m);
  wrap_field(m);
  wrap_grid(m);
  wrap_quantity(m);
  wrap_timesolver(m);
  wrap_variable(m);
  wrap_world(m);
}