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
  real maxerror() const;

  void adaptTimeStep(real corr);
  void setTime(real);

  void step();
  void steps(int);

 private:
  real maxerror_;
  real time_;
  real dt_;
  DynamicEquation eq_;
  Stepper* stepper_;
};