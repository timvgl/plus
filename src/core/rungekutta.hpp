#pragma once

#include <memory>
#include <vector>

#include "butchertableau.hpp"
#include "dynamicequation.hpp"
#include "stepper.hpp"

class TimeSolver;
class Field;
class RungeKuttaStageExecutor;

class RungeKuttaStepper : public Stepper {
 public:
  RungeKuttaStepper(TimeSolver*, RKmethod);
  int nStages() const;
  void step();

 private:
  ButcherTableau butcher_;

  friend class RungeKuttaStageExecutor;
};

class RungeKuttaStageExecutor {
 public:
  RungeKuttaStageExecutor(DynamicEquation eq, RungeKuttaStepper* stepper);

  void setStageK(int stage);
  void setStageX(int stage);
  void setFinalX();
  void resetX();
  real getError();

 private:
  std::unique_ptr<Field> noise_;
  std::unique_ptr<Field> x0_;
  std::vector<std::unique_ptr<Field>> k_;
  DynamicEquation eq_;
  RungeKuttaStepper* stepper_;
};