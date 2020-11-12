#include "linsolver.hpp"

void JacobiStepper::step() {
  Field r = sys_->residual(x_);  // r = Ax-b
  x_ = add(1.0, x_, -1.0, r);    // x = x-r
}

Field solveLinearSystem(const LinearSystem& sys) {
  JacobiStepper stepper(&sys);
  for (int i = 0; i < 1000; i++) {
    stepper.step();
  }
  return stepper.x();
}
