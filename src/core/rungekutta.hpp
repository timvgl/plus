#pragma once

#include "stepper.hpp"
#include "butchertableau.hpp"

#include<vector>
#include<memory>

class TimeSolver;
class Field;

class RungeKuttaStepper : public Stepper {
 public:
  RungeKuttaStepper(TimeSolver*, RKmethod);
  void step();
 private:
  ButcherTableau butcher_;
  std::vector<std::unique_ptr<Field>> k_;
};