#include "timesolver.hpp"

#include "eulerstepper.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

TimeSolver::TimeSolver(DynamicEquation eq, real timestep)
    : time_(0), dt_(timestep), eq_(eq) {
  stepper_ = new RungeKuttaStepper(this,DORMANDPRINCE);
  //stepper_ = new EulerStepper(this);
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

void TimeSolver::setTime(real time) {
  time_ = time;
}

void TimeSolver::step() {
  stepper_->step();
}