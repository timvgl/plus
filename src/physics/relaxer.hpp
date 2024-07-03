#pragma once

#include "ferromagnetquantity.hpp"
#include "timesolver.hpp"

class Magnet;

// Relax the system to a minimum energy state by first minimizing the total energy
// and then minimizing the total torque.

class Relaxer {
 public:
  Relaxer(const Magnet*, real RelaxTorqueThreshold);
  
  void exec();
  std::vector<FM_FieldQuantity> getTorque();

 private:
  const Magnet* magnet_;
  real threshold_;
};