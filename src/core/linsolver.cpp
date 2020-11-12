#include "linsolver.hpp"

void JacobiStepper::step() {
  Field r = sys_->residual(*x_);  // r = Ax-b
  *x_ = add(1.0, *x_, -1.0, r);   // x = x-r
}