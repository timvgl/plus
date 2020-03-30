#pragma once
#include "datatypes.hpp"
#include "dynamicequation.hpp"

class Stepper;
class Variable;
class Quantity;

class TimeSolver {
 public:
  TimeSolver(DynamicEquation eq, real timestep);
  ~TimeSolver();

  real time() const;
  DynamicEquation eq() const;
  real timestep() const;

  void step();

 private:
  real time_;
  real dt_;
  DynamicEquation eq_;
  Stepper* stepper_;
};