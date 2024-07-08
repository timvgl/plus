#pragma once

#include "ferromagnetquantity.hpp"
#include "timesolver.hpp"

class Magnet;

// Relax the system to a minimum energy state by first minimizing the total energy
// and then minimizing the total torque.

class Relaxer {
 public:
  Relaxer(const Magnet*, std::vector<real> RelaxTorqueThreshold);
  
  void exec();

  // Helper functions to execute relax procedure
  std::vector<FM_FieldQuantity> getTorque();
  real calcTorque(std::vector<FM_FieldQuantity>);

 private:
  const Magnet* magnet_;
  std::vector<real> threshold_;
};