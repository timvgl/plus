#include <iostream>

#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "timesolver.hpp"
#include "world.hpp"

int main() {
  World world({1.0, 1.0, 1.0});
  Ferromagnet* magnet = world.addFerromagnet("my_magnet", Grid({64, 64, 1}));
  DynamicEquation llg(magnet->magnetization(), magnet->torque());
  TimeSolver solver(llg, 1e-2);
  solver.step();
  return 0;
}