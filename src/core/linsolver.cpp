#include "linsolver.hpp"

#include "field.hpp"
#include "fieldops.hpp"
#include "linsystem.hpp"
#include "reduce.hpp"

void JacobiStepper::step() {
  Field r = sys_->residual(x_);  // r = b-Ax
  x_ = add(x_, r);               // x = x+r
}

void ConjugateGradientStepper::step() {
  Field Ap = sys_->matrixmul(p);    // Ap  = A*p
  real alpha = rr / dotSum(p, Ap);  // α   = rr/(p,Ap)
  x_ = add(1.0, x_, alpha, p);      // x   = x + α*p
  r = add(1.0, r, -alpha, Ap);      // r   = r - α*Ap
  real rrPrime = rr;                // rr' = rr
  rr = dotSum(r, r);                // rr  = (r,r)
  real beta = rr / rrPrime;         // β   = rr/rr'
  p = add(1.0, r, beta, p);         // p   = r + β*p
}

void ConjugateGradientStepper::restart() {
  r = sys_->residual(x_);  // r = b-A*x
  p = r;                   // p = r
  rr = dotSum(r, r);       // rr = (r,r)
}