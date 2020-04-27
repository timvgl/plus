#pragma once
#include <functional>

#include "datatypes.hpp"
#include "dynamicequation.hpp"

class Stepper;
class Variable;
class FieldQuantity;

class TimeSolver {
 public:
  TimeSolver(DynamicEquation eq);
  ~TimeSolver();

  DynamicEquation eq() const;
  real time() const;
  real timestep() const;
  bool adaptiveTimeStep() const;
  real maxerror() const;

  void adaptTimeStep(real corr);
  void setTime(real);
  void setTimeStep(real);
  void enableAdaptiveTimeStep(); // default
  void disableAdaptiveTimeStep();

  void step();
  void steps(unsigned int nsteps);
  void runwhile(std::function<bool(void)>);
  void run(real duration);

 private:
  real maxerror_;
  real time_;
  real timestep_;
  bool fixedTimeStep_;
  DynamicEquation eq_;
  Stepper* stepper_;
};