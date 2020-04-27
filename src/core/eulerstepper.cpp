#include "eulerstepper.hpp"

#include <memory>

#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"

EulerStepper::EulerStepper(TimeSolver* solver) {
  setParentTimeSolver(solver);
}

void EulerStepper::step() {
  auto eq = solver_->eq();
  auto x = eq.x->field();
  auto dxdt = eq.rhs->eval();
  auto xnew = std::unique_ptr<Field>(new Field(eq.x->grid(), eq.x->ncomp()));
  real dt = solver_->timestep();

  add(xnew.get(), 1.0, x, dt, dxdt.get());  // xnew = x + dt*dxdt

  eq.x->set(xnew.get());
  solver_->setTime(solver_->time() + dt);
}