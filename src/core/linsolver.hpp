#pragma once

#include "field.hpp"
#include "fieldops.hpp"
#include "linsystem.hpp"

class LinearSystemSolverStepper {
 public:
  LinearSystemSolverStepper(const LinearSystem* sys)
      : sys_(sys), x_(sys->grid(), 1) {}
  virtual void step() = 0;
  Field x() const { return x_; }
  void setx(const Field& x) { x_ = x; }

 protected:
  const LinearSystem* sys_;
  Field x_;
};

class JacobiStepper : public LinearSystemSolverStepper {
 public:
  JacobiStepper(const LinearSystem* sys) : LinearSystemSolverStepper(sys) {}
  void step();
};

Field solveLinearSystem(const LinearSystem& sys);