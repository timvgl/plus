#include "timesolver.hpp"

#include <cmath>

#include "eulerstepper.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

TimeSolver::TimeSolver(DynamicEquation eq, real timestep)
    : time_(0), dt_(timestep), maxerror_(1e-5), eq_(eq) {
  stepper_ = new RungeKuttaStepper(this, FEHLBERG);
  // stepper_ = new EulerStepper(this);
}

TimeSolver::~TimeSolver() {
  if (stepper_)
    delete stepper_;
}

real TimeSolver::time() const {
  return time_;
}

DynamicEquation TimeSolver::eq() const {
  return eq_;
}

real TimeSolver::timestep() const {
  return dt_;
}

real TimeSolver::maxerror() const {
  return maxerror_;
}

void TimeSolver::setTime(real time) {
  time_ = time;
}

void TimeSolver::adaptTimeStep(real corr) {
  real headroom = 0.8;

  if (std::isnan(corr))
    corr = 1.;
  corr *= headroom;
  corr = corr > 2.0 ? 2.0 : corr;
  corr = corr < 0.5 ? 0.5 : corr;
  dt_ *= corr;
}

void TimeSolver::step() {
  stepper_->step();
}

void TimeSolver::steps(int nSteps) {
  for (int i = 0; i < nSteps; i++)
    step();
}