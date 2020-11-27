#pragma once

#include <deque>

#include "ferromagnetquantity.hpp"

class Ferromagnet;

// Minimize follows the steepest descent method as per Exl et al., JAP 115,
// 17D118 (2014).

class Minimizer {
 public:
  Minimizer(const Ferromagnet*, real stopMaxMagDiff, int nMagDiffSamples);

  void exec();

 private:
  void step();
  const Ferromagnet* magnet_;
  real stepsize_;
  int nsteps_;

  FM_FieldQuantity torque_;
  Field t0, t1, m0, m1;

  real stopMaxMagDiff_;
  int nMagDiffSamples_;
  bool converged() const;
  void addMagDiff(real);
  std::deque<real> lastMagDiffs_;
};
