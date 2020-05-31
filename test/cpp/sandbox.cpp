#include <iostream>
#include <memory>

#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include "timesolver.hpp"
#include "world.hpp"
#include "variable.hpp"
#include "torque.hpp"


int main() {
  World world({4e-9, 4e-9, 4e-9});
  Ferromagnet* magnet = world.addFerromagnet(Grid({128, 32, 1}));
  magnet->msat.set(800e3);
  magnet->aex.set(13e-12);
  magnet->alpha.set(0.02);
  magnet->magnetization()->set({1,0.1,0});
  magnet->minimize();

  std::shared_ptr<FieldQuantity> torque(torqueQuantity(magnet).clone());

  DynamicEquation llg(magnet->magnetization(), torque);


  TimeSolver solver(llg);
  solver.step();

  return 0;
}