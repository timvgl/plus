#pragma once

#include "ferromagnetquantity.hpp"
#include "timesolver.hpp"

class DynamicEquation;
class Magnet;
class MumaxWorld;
class TimeSolver;
// Relax the system to a minimum energy state by first minimizing the total energy
// and then minimizing the total torque.

class Relaxer {
 public:
  Relaxer(const Magnet*, std::vector<real> RelaxTorqueThreshold);
  Relaxer(const MumaxWorld*, real RelaxTorqueThreshold);
  
  // Execute relaxing procedure
  void exec();

  // Helper functions
 private:
  std::vector<DynamicEquation> getEquation(const Magnet*);
  std::vector<FM_FieldQuantity> getTorque();
  real calcTorque(std::vector<FM_FieldQuantity>);
  real calcEnergy();

 private:
  std::vector<const Magnet*> magnets_;
  std::vector<real> threshold_;
  TimeSolver &timesolver_;
  const MumaxWorld* world_;
};