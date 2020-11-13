#pragma once

#include "field.hpp"

class LinearSystem;

class LinearSystemSolverStepper {
 public:
  LinearSystemSolverStepper(const LinearSystem* sys, Field* x)
      : sys_(sys), x_(*x) {}
  virtual void step() = 0;
  virtual void restart() {}

 protected:
  const LinearSystem* sys_;
  Field& x_;
};

class JacobiStepper : public LinearSystemSolverStepper {
 public:
  JacobiStepper(const LinearSystem* sys, Field* x)
      : LinearSystemSolverStepper(sys, x) {}
  void step();
};

class ConjugateGradientStepper : public LinearSystemSolverStepper {
 public:
  ConjugateGradientStepper(const LinearSystem* sys, Field* x)
      : LinearSystemSolverStepper(sys, x) {}
  void step();
  void restart();

 private:
  real rr;
  Field p;
  Field r;
};