#pragma once

#include "stepper.hpp"

class TimeSolver;

class EulerStepper : public Stepper {
 public:
  EulerStepper(TimeSolver*);
  void step();
};