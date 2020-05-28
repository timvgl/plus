#pragma once

#include <deque>

#include "torque.hpp"

class Ferromagnet;

// Minimize follows the steepest descent method as per Exl et al., JAP 115,
// 17D118 (2014).

class Minimizer {
 public:
  Minimizer(Handle<Ferromagnet>, real stopMaxMagDiff, int nMagDiffSamples);
  ~Minimizer();

  void exec();

 private:
  void step();
  Handle<Ferromagnet> magnet_;
  real stepsize_;
  int nsteps_;

  RelaxTorque torque_;
  Field *t_old, *t_new, *m_old, *m_new;

  real stopMaxMagDiff_;
  int nMagDiffSamples_;
  bool converged() const;
  void addMagDiff(real);
  std::deque<real> lastMagDiffs_;
};