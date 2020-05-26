#include "rungekutta.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "butchertableau.hpp"
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
  std::vector<RungeKuttaStageExecutor> equations;
  for (int i = 0; i < solver_->nEquations(); i++)
    equations.emplace_back(RungeKuttaStageExecutor(solver_->equation(i), this));

  real dt = solver_->timestep();
  real t0 = solver_->time();

  // Apply the stages
  for (int stage = 0; stage < nStages(); stage++) {
    solver_->setTime(t0 + dt * butcher_.nodes[stage]);

    for (auto& eq : equations)
      eq.setStageX(stage);

    for (auto& eq : equations)
      eq.setStageK(stage);
  }

  // Make the actual step
  solver_->setTime(t0 + dt);
  for (auto& eq : equations)
    eq.setFinalX();

  // determine the error
  real err = 0.0;
  for (auto& eq : equations) {
    real eqerr = eq.getError();
    if (eqerr > err)
      err = eqerr;
  }

  // update the timestep
  real corr;
  if (err < solver_->maxerror()) {
    corr = std::pow(solver_->maxerror() / err, 1. / butcher_.order2);
  } else {
    corr = std::pow(solver_->maxerror() / err, 1. / butcher_.order1);
  }
  solver_->adaptTimeStep(corr);

  // undo step if error exceeds the tolerance
  if (err > solver_->maxerror()) {
    for (auto& eq : equations)
      eq.resetX();
    solver_->setTime(t0);
  }
}

RungeKuttaStageExecutor::RungeKuttaStageExecutor(DynamicEquation eq,
                       RungeKuttaStepper* stepper)
    : eq_(eq), x0_(new Field(eq.grid(), eq.ncomp())), stepper_(stepper) {

  int nStages = stepper_->nStages();
  k_.reserve(nStages);
  for (int stage = 0; stage < nStages; stage++) {
    k_.emplace_back(new Field(eq.grid(), eq.ncomp()));
  }

  x0_->copyFrom(eq_.x->field());
}

void RungeKuttaStageExecutor::setStageK(int stage) {
  eq_.rhs->evalIn(k_[stage].get());
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