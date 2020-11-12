#pragma once

#include "field.hpp"
#include "fieldops.hpp"
#include "linsystem.hpp"

class LinearSystemSolverStepper {
 public:
  LinearSystemSolverStepper(const LinearSystem* sys, Field* x)
      : sys_(sys), x_(x) {}
  virtual void step() = 0;

 protected:
  const LinearSystem* sys_;
  Field* x_;
};

class JacobiStepper : public LinearSystemSolverStepper {
 public:
  JacobiStepper(const LinearSystem* sys, Field* x)
      : LinearSystemSolverStepper(sys, x) {}
  void step();
};