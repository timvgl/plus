#pragma once

class TimeSolver;

class Stepper {
 public:
  virtual void step() = 0;
  void setParentTimeSolver(TimeSolver*);

 protected:
  TimeSolver* solver_;
};