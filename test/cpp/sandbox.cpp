#include <iostream>

#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "timesolver.hpp"
#include "world.hpp"

int main() {
  World world({1.0, 1.0, 1.0});
  Ferromagnet* magnet = world.addFerromagnet("my_magnet", Grid({8, 8, 1}));
  DynamicEquation llg(magnet->magnetization(), magnet->torque());
  TimeSolver solver(llg, 1e-9);
  for (int i = 0; i < 1410; i++) {
      std::cout << i << std::endl;
    solver.step();
  }

  for (int i = 0; i < 1414; i++) {
      std::cout << i << std::endl;
    solver.step();
  }

  return 0;
}