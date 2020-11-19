#pragma once

#include "field.hpp"

class LinearSystem;

class LinearSystemSolverStepper {
 public:
  enum Method { JACOBI, CONJUGATEGRADIENT, MINIMALRESIDUAL, STEEPESTDESCENT };
  static Method getMethodByName(std::string);

 public:
  LinearSystemSolverStepper(const LinearSystem* sys, Field* x)
      : sys_(sys), x_(*x) {}
  virtual void step() = 0;
  virtual void restart() {}

 public:
  // factory method to create a stepper for the given method
  static std::unique_ptr<LinearSystemSolverStepper> create(LinearSystem*,
                                                           Field*,
                                                           Method);

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

class MinimalResidualStepper : public LinearSystemSolverStepper {
 public:
  MinimalResidualStepper(const LinearSystem* sys, Field* x)
      : LinearSystemSolverStepper(sys, x) {}
  void step();
  void restart();

 private:
  real rr;
  Field p;
  Field r;
};

class SteepestDescentStepper : public LinearSystemSolverStepper {
 public:
  SteepestDescentStepper(const LinearSystem* sys, Field* x)
      : LinearSystemSolverStepper(sys, x) {}
  void step();
  void restart();

 private:
  real rr;
  Field p;
  Field r;
};