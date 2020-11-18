#include "linsolver.hpp"

#include <memory>
#include <stdexcept>

#include "field.hpp"
#include "fieldops.hpp"
#include "linsystem.hpp"
#include "reduce.hpp"

LinearSystemSolverStepper::Method LinearSystemSolverStepper::getMethodByName(
    std::string methodName) {
  if (methodName == "jacobi")
    return Method::JACOBI;
  if (methodName == "conjugategradient")
    return Method::CONJUGATEGRADIENT;
  if (methodName == "minimalresidual")
    return Method::MINIMALRESIDUAL;
  if (methodName == "steepestdescent")
    return Method::STEEPESTDESCENT;
  throw std::invalid_argument("Linear system solver method '" + methodName +
                              "' is not implemented");
}

std::unique_ptr<LinearSystemSolverStepper>
LinearSystemSolverStepper::create(LinearSystem* sys, Field* x, Method method) {
  switch (method) {
    case JACOBI:
      return std::make_unique<JacobiStepper>(sys, x);
    case CONJUGATEGRADIENT:
      return std::make_unique<ConjugateGradientStepper>(sys, x);
    case MINIMALRESIDUAL:
      return std::make_unique<MinimalResidualStepper>(sys, x);
    case STEEPESTDESCENT:
      return std::make_unique<SteepestDescentStepper>(sys, x);
  }
}

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

void MinimalResidualStepper::step() {
  real pp = dotSum(p, p);       // pp = (p,p)
  real alpha = rr / pp;         // α  = rr/pp
  x_ = add(1.0, x_, alpha, r);  // x  = x + α*r
  r = add(1.0, r, -alpha, p);   // r  = r - α*p
  p = sys_->matrixmul(r);       // p  = A*r
  rr = dotSum(r, r);            // rr = (r,r)
}

void MinimalResidualStepper::restart() {
  r = sys_->residual(x_);  // r  = b-A*x
  p = sys_->matrixmul(r);  // p  = A*r
  rr = dotSum(r, r);       // rr = (r,r)
}

void SteepestDescentStepper::step() {
  real pr = dotSum(p, r);       // pr = (p,r)
  real alpha = rr / pr;         // α  = rr/pr
  x_ = add(1.0, x_, alpha, r);  // x  = x + α*r
  r = add(1.0, r, -alpha, p);   // r  = r - α*p
  p = sys_->matrixmul(r);       // p  = A*r
  rr = dotSum(r, r);            // rr = (r,r)
}

void SteepestDescentStepper::restart() {
  r = sys_->residual(x_);  // r  = b-A*x
  p = sys_->matrixmul(r);  // p  = A*r
  rr = dotSum(r, r);       // rr = (r,r)
}