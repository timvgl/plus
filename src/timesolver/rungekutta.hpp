#pragma once

#include <optional>

#include "butchertableau.hpp"
#include "field.hpp"
#include "dynamicequation.hpp"
#include "stepper.hpp"

class TimeSolver;
class Variable;

class RungeKuttaStepper : public Stepper {
 public:
  RungeKuttaStepper(TimeSolver*, RKmethod);
  int nStages() const;
  void step();

 private:
  ButcherTableau butcher_;
  class RungeKuttaStageExecutor;
};

class RungeKuttaStepper::RungeKuttaStageExecutor {
  public:
    RungeKuttaStageExecutor(DynamicEquation eq, RungeKuttaStepper* stepper);

    void setStageK(int stage);
    void setStageX(int stage);
    void setFinalX();
    void resetX();
    real getError() const;

  private:
    Field x0;
    const ButcherTableau butcher;
    const real& dt;
    const Variable& x;  // TODO: make this non constant
    std::optional<Field> noise;
    std::vector<Field> k;
    DynamicEquation eq_;
};

