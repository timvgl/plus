#include "rungekutta.hpp"

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

struct RKEquation {
  RKEquation(DynamicEquation eq, int nStages)
      : x(eq.x),
        rhs(eq.rhs),
        x0(new Field(eq.grid(), eq.ncomp())),
        xstage(new Field(eq.grid(), eq.ncomp())) {
    k.reserve(nStages);
    for (int stage = 0; stage < nStages; stage++) {
      k.emplace_back(new Field(eq.grid(), eq.ncomp()));
    }
  }
  const Variable* x;
  const FieldQuantity* rhs;
  std::unique_ptr<Field> x0;
  std::unique_ptr<Field> xstage;
  std::vector<std::unique_ptr<Field>> k;
};

RungeKuttaStepper::RungeKuttaStepper(TimeSolver* solver, RKmethod method)
    : butcher_(constructTableau(method)) {
  setParentTimeSolver(solver);

  // TODO: fix this for multiple equations
  DynamicEquation eq = solver_->equation(0);

  k_.reserve(butcher_.nStages);
  for (int stage = 0; stage < butcher_.nStages; stage++) {
    k_.emplace_back(new Field(eq.grid(), eq.ncomp()));
  }
}

int RungeKuttaStepper::nStages() const {
  return butcher_.nStages;
}

void RungeKuttaStepper::step() {
  std::vector<RKEquation> equations;
  equations.reserve(solver_->nEquations());
  for (int i = 0; i < solver_->nEquations(); i++) {
    equations.emplace_back(RKEquation(solver_->equation(i), nStages()));
  }

  real dt = solver_->timestep();
  real t0 = solver_->time();

  for (auto& eq : equations)
    eq.x0->copyFrom(eq.x->field());

  // Apply the stages
  for (int stage = 0; stage < butcher_.nStages; stage++) {
    for (auto& eq : equations) {
      solver_->setTime(t0 + dt * butcher_.nodes[stage]);

      if (stage > 0) {
        std::vector<real> weights(1 + stage);
        std::vector<const Field*> fields(1 + stage);
        weights[0] = 1;
        fields[0] = eq.x0.get();
        for (int i = 0; i < stage; i++) {
          weights[i + 1] = dt * butcher_.rkMatrix[stage][i];
          fields[i + 1] = k_[i].get();
        }
        add(eq.xstage.get(), fields, weights);
        eq.x->set(eq.xstage.get());
      }

      eq.rhs->evalIn(k_[stage].get());
    }
  }

  // Make the actual step
  for (auto& eq : equations) {
    std::vector<real> weights(1 + butcher_.nStages);
    std::vector<const Field*> fields(1 + butcher_.nStages);
    weights[0] = 1;
    fields[0] = eq.x0.get();
    for (int i = 0; i < butcher_.nStages; i++) {
      weights[i + 1] = dt * butcher_.weights1[i];
      fields[i + 1] = k_[i].get();
    }
    add(eq.xstage.get(), fields, weights);
    eq.x->set(eq.xstage.get());
  }
  solver_->setTime(t0 + dt);

  // Determine the error
  real err = 0.0;
  for (auto& eq : equations) {
    std::vector<real> weights_err(butcher_.nStages);
    std::vector<const Field*> fields_err(butcher_.nStages);
    for (int i = 0; i < butcher_.nStages; i++) {
      fields_err[i] = k_[i].get();
      weights_err[i] = dt * (butcher_.weights1[i] - butcher_.weights2[i]);
    }
    add(eq.xstage.get(), fields_err, weights_err);
    real eqerr = maxVecNorm(eq.xstage.get());
    if (eqerr > err)
      err = eqerr;
  }

  real corr;
  if (err < solver_->maxerror()) {
    corr = std::pow(solver_->maxerror() / err, 1. / butcher_.order2);
    solver_->adaptTimeStep(corr);
  } else {
    corr = std::pow(solver_->maxerror() / err, 1. / butcher_.order1);
    solver_->adaptTimeStep(corr);
    // undo step
    for (auto& eq : equations) {
      eq.x->set(eq.x0.get());
    }
    solver_->setTime(t0);
  }
}