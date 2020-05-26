#pragma once

#include <memory>
#include <vector>

#include "butchertableau.hpp"
#include "stepper.hpp"

class TimeSolver;
class Field;

class RungeKuttaStepper : public Stepper {
 public:
  RungeKuttaStepper(TimeSolver*, RKmethod);
  int nStages() const;
  void step();

 private:
  ButcherTableau butcher_;
};