#include "rungekutta.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "butchertableau.hpp"
#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "timesolver.hpp"
#include "variable.hpp"

RungeKuttaStepper::RungeKuttaStepper(TimeSolver* solver, RKmethod method)
    : butcher_(constructTableau(method)) {
  setParentTimeSolver(solver);
}

int RungeKuttaStepper::nStages() const {
  return butcher_.nStages;
}

void RungeKuttaStepper::step() {
  // construct a Runge Kutta stage executor for every equation
  std::vector<RungeKuttaStageExecutor> equations;
  for (int i = 0; i < solver_->nEquations(); i++)
    equations.emplace_back(RungeKuttaStageExecutor(solver_->equation(i), this));

  real t0 = solver_->time();

  bool success = false;
  while (!success) {
    real dt = solver_->timestep();

    // apply the stages
    for (int stage = 0; stage < nStages(); stage++) {
      solver_->setTime(t0 + dt * butcher_.nodes[stage]);
      for (auto& eq : equations)
        eq.setStageX(stage);
      for (auto& eq : equations)
        eq.setStageK(stage);
    }

    // make the actual step
    solver_->setTime(t0 + dt);
    for (auto& eq : equations)
      eq.setFinalX();

    // nothing more to do if time step is fixed
    if (!solver_->adaptiveTimeStep())
      break;

    // loop over equations and get the largest error
    real error = 0.0;
    for (auto& eq : equations)
      if (real e = eq.getError(); e > error)
        error = e;

    success = error < solver_->maxerror();

    // update the timestep
    real corrFactor;
    if (success) {
      corrFactor = std::pow(solver_->maxerror() / error, 1. / butcher_.order2);
    } else {
      corrFactor = std::pow(solver_->maxerror() / error, 1. / butcher_.order1);
    }
    solver_->adaptTimeStep(corrFactor);

    // undo step if not successful
    if (!success) {
      for (auto& eq : equations)
        eq.resetX();
      solver_->setTime(t0);
    }
  }
}

RungeKuttaStageExecutor::RungeKuttaStageExecutor(DynamicEquation eq,
                                                 RungeKuttaStepper* stepper)
    : eq_(eq), stepper_(stepper), k_(stepper->nStages()) {
  // Noise term evaluated only here, it remains constant throughout all stages
  if (eq_.noiseTerm && !eq_.noiseTerm->assuredZero())
    noise_ = eq_.noiseTerm->eval();

  // make back up of the x value
  x0_ = eq_.x->field()->newCopy();
}

void RungeKuttaStageExecutor::setStageK(int stage) {
  k_[stage] = eq_.rhs->eval();

  if (noise_) {
    real dt = stepper_->solver_->timestep();

    // k += sqrt(1/dt)*noise
    add(k_[stage].get(), 1.0, k_[stage].get(), 1 / sqrt(dt), noise_.get());
  }
}

void RungeKuttaStageExecutor::setStageX(int stage) {
  if (stage == 0)
    return;

  real dt = stepper_->solver_->timestep();

  std::vector<real> weights(1 + stage);
  std::vector<const Field*> fields(1 + stage);

  weights[0] = 1;
  fields[0] = x0_.get();
  for (int i = 0; i < stage; i++) {
    weights[i + 1] = dt * stepper_->butcher_.rkMatrix[stage][i];
    fields[i + 1] = k_[i].get();
  }

  std::unique_ptr<Field> xstage(new Field(eq_.grid(), eq_.ncomp()));
  add(xstage.get(), fields, weights);
  eq_.x->set(xstage.get());
}

void RungeKuttaStageExecutor::setFinalX() {
  real dt = stepper_->solver_->timestep();
  auto butcher = stepper_->butcher_;

  std::vector<real> weights(1 + butcher.nStages);
  std::vector<const Field*> fields(1 + butcher.nStages);

  weights[0] = 1;
  fields[0] = x0_.get();
  for (int i = 0; i < butcher.nStages; i++) {
    weights[i + 1] = dt * butcher.weights1[i];
    fields[i + 1] = k_[i].get();
  }

  std::unique_ptr<Field> xstage(new Field(eq_.grid(), eq_.ncomp()));
  add(xstage.get(), fields, weights);
  eq_.x->set(xstage.get());
}

void RungeKuttaStageExecutor::resetX() {
  eq_.x->set(x0_.get());
}

real RungeKuttaStageExecutor::getError() {
  real dt = stepper_->solver_->timestep();
  auto butcher = stepper_->butcher_;

  std::vector<real> weights_err(butcher.nStages);
  std::vector<const Field*> fields_err(butcher.nStages);

  for (int i = 0; i < butcher.nStages; i++) {
    fields_err[i] = k_[i].get();
    weights_err[i] = dt * (butcher.weights1[i] - butcher.weights2[i]);
  }

  std::unique_ptr<Field> xstage(new Field(eq_.grid(), eq_.ncomp()));
  add(xstage.get(), fields_err, weights_err);
  return maxVecNorm(xstage.get());
}