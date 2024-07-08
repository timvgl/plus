#pragma once

#include "ferromagnetquantity.hpp"
#include "timesolver.hpp"

class Magnet;
class MumaxWorld;
class TimeSolver;
// Relax the system to a minimum energy state by first minimizing the total energy
// and then minimizing the total torque.

class Relaxer {
 public:
  Relaxer(const Magnet*, std::vector<real> RelaxTorqueThreshold);
  Relaxer(const MumaxWorld*);
  
  void exec();
  std::vector<FM_FieldQuantity> getTorque();
  real calcEnergy();
  bool converged(std::vector<real> t1, std::vector<real> t2);

 private:
  std::vector<const Magnet*> magnets_;
  std::vector<real> threshold_;
  const MumaxWorld* world_;
  TimeSolver &timesolver_;
};