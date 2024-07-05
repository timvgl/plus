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
  std::vector<FM_FieldQuantity> getTorque();
  bool converged(std::vector<real> t1, std::vector<real> t2);

 private:
  const Magnet* magnet_;
  std::vector<real> threshold_;
};