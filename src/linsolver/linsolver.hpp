#pragma once

#include <memory>
#include <string>

#include "field.hpp"
#include "linsystem.hpp"

class LinSolverStepper;

class LinSolver {
 public:
  enum Method { JACOBI, CONJUGATEGRADIENT, MINIMALRESIDUAL, STEEPESTDESCENT };
  static Method getMethodByName(std::string);

 public:
  LinSolver(std::unique_ptr<LinearSystem> system);
  void setSystem(std::unique_ptr<LinearSystem> newSystem);
  void setMethod(Method);
  void setMethodByName(std::string);
  void resetState();
  Field getState() const;
  void setState(const Field& newx);
  Field solve();
  Field residual() const;
  real residualMaxNorm() const;

  void step();
  void restartStepper();

 public:
  int maxIterations = -1;
  double tol = 1e-5;

 private:
  std::unique_ptr<LinSolverStepper> stepper_;
  std::unique_ptr<LinearSystem> system_;
  Field x_;

  friend LinSolverStepper;
};

class LinSolverStepper {
 public:
  using Method = LinSolver::Method;
  static std::unique_ptr<LinSolverStepper> create(LinSolver* parent, Method);

 public:
  LinSolverStepper(LinSolver* parent) : parent_(parent) {}
  virtual Method method() const = 0;
  virtual void step() = 0;
  virtual void restart(){};

 protected:
  LinearSystem* system() const { return parent_->system_.get(); }
  Field& xRef() const { return parent_->x_; }

 private:
  LinSolver* parent_;
};