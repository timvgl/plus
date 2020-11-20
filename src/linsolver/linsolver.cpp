#include "linsolver.hpp"

#include <memory>

#include "fieldops.hpp"
#include "reduce.hpp"

LinSolver::LinSolver(std::unique_ptr<LinearSystem> system) {
  stepper_ = LinSolverStepper::create(this, JACOBI);
  setSystem(std::move(system));
}

void LinSolver::setSystem(std::unique_ptr<LinearSystem> newSystem) {
  system_ = std::move(newSystem);
  if (system_->grid() != x_.grid()) {
    resetState();
  }
}

void LinSolver::setMethod(Method method) {
  stepper_ = LinSolverStepper::create(this, method);
  restartStepper();
}

void LinSolver::setMethodByName(std::string methodname) {
  setMethod(getMethodByName(methodname));
}

void LinSolver::resetState() {
  x_ = Field(system_->grid(), 1);
  restartStepper();
}

Field LinSolver::getState() const {
  return x_;
}

void LinSolver::setState(const Field& newx) {
  // todo check grid and ncomp
  x_ = newx;
}

Field LinSolver::solve() {
  int nstep = 0;
  while ((double)residualMaxNorm() > tol) {
    if (nstep > maxIterations && maxIterations >= 0) {
      break;
    }

    step();
    nstep++;
  }
  return x_;
}

void LinSolver::step() {
  stepper_->step();
}

void LinSolver::restartStepper() {
  stepper_->restart();
}

Field LinSolver::residual() const {
  return system_->residual(x_);
}

real LinSolver::residualMaxNorm() const {
  return maxAbsValue(system_->residual(x_));
}

//--- IMPLEMENTATION OF STEPPERS -----------------------------------
// These implementations should only be used in this compilation unit,
// hence the anonymous namespace
namespace {

class Jacobi : public LinSolverStepper {
 public:
  Jacobi(LinSolver* parent) : LinSolverStepper(parent) {}
  Method method() const { return Method::JACOBI; }

  void step() {
    Field& x = xRef();
    Field r = system()->residual(x);  // r = b-Ax
    x = add(x, r);                    // x = x+r
  }
};

class ConjgentGradient : public LinSolverStepper {
 public:
  ConjgentGradient(LinSolver* parent) : LinSolverStepper(parent) {}
  Method method() const { return Method::CONJUGATEGRADIENT; }

  void step() {
    Field& x = xRef();
    Field Ap = system()->matrixmul(p);  // Ap  = A*p
    real alpha = rr / dotSum(p, Ap);    // α   = rr/(p,Ap)
    x = add(1.0, x, alpha, p);          // x   = x + α*p
    r = add(1.0, r, -alpha, Ap);        // r   = r - α*Ap
    real rrPrime = rr;                  // rr' = rr
    rr = dotSum(r, r);                  // rr  = (r,r)
    real beta = rr / rrPrime;           // β   = rr/rr'
    p = add(1.0, r, beta, p);           // p   = r + β*p
  }

  void restart() {
    Field& x = xRef();
    r = system()->residual(x);  // r = b-A*x
    p = r;                      // p = r
    rr = dotSum(r, r);          // rr = (r,r)
  }

 private:
  real rr;
  Field p;
  Field r;
};

class MinimalResidual : public LinSolverStepper {
 public:
  MinimalResidual(LinSolver* parent) : LinSolverStepper(parent) {}
  Method method() const { return Method::MINIMALRESIDUAL; }

  void step() {
    Field& x = xRef();
    real pp = dotSum(p, p);      // pp = (p,p)
    real alpha = rr / pp;        // α  = rr/pp
    x = add(1.0, x, alpha, r);   // x  = x + α*r
    r = add(1.0, r, -alpha, p);  // r  = r - α*p
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

  void restart() {
    Field& x = xRef();
    r = system()->residual(x);   // r  = b-A*x
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

 private:
  real rr;
  Field p;
  Field r;
};

class SteepestDescent : public LinSolverStepper {
 public:
  SteepestDescent(LinSolver* parent) : LinSolverStepper(parent) {}
  Method method() const { return Method::STEEPESTDESCENT; }

  void step() {
    Field& x = xRef();
    real pr = dotSum(p, r);      // pr = (p,r)
    real alpha = rr / pr;        // α  = rr/pr
    x = add(1.0, x, alpha, r);   // x  = x + α*r
    r = add(1.0, r, -alpha, p);  // r  = r - α*p
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

  void restart() {
    Field& x = xRef();
    r = system()->residual(x);   // r  = b-A*x
    p = system()->matrixmul(r);  // p  = A*r
    rr = dotSum(r, r);           // rr = (r,r)
  }

 private:
  real rr;
  Field p;
  Field r;
};

}  // namespace

//--- LINSOLVER FACTORT METHOD -----------------------------------

std::unique_ptr<LinSolverStepper> LinSolverStepper::create(LinSolver* parent,
                                                           Method method) {
  switch (method) {
    case Method::JACOBI:
      return std::make_unique<Jacobi>(parent);
    case Method::CONJUGATEGRADIENT:
      return std::make_unique<ConjgentGradient>(parent);
    case Method::MINIMALRESIDUAL:
      return std::make_unique<MinimalResidual>(parent);
    case Method::STEEPESTDESCENT:
      return std::make_unique<SteepestDescent>(parent);
  }
}

LinSolver::Method LinSolver::getMethodByName(std::string methodName) {
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