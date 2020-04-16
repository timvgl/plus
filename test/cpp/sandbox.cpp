#include <iostream>

#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include "timesolver.hpp"
#include "world.hpp"

int main() {
  World world({1.0, 1.0, 1.0});
  Ferromagnet* magnet = world.addFerromagnet("my_magnet", Grid({128, 64, 1}));
  DynamicEquation llg(magnet->magnetization(), magnet->torque());
  TimeSolver solver(llg, 1e-2);

  solver.steps(1000);

  timer.printTimings();

  return 0;
}