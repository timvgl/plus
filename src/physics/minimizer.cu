#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "minimizer.hpp"
#include "reduce.hpp"
#include "torque.hpp"

#include <iostream>

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : magnet_({magnet}),
      torque_({relaxTorqueQuantity(magnet)}),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      t0(magnet_.size()),
      t1(magnet_.size()),
      m0(magnet_.size()),
      m1(magnet_.size()) {
  stepsize_ = {1e-14};  // TODO: figure out how to make descent guess
  // TODO: check if input arguments are sane
}

Minimizer::Minimizer(const Antiferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : magnet_(magnet->sublattices()),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      t0(magnet_.size()),
      t1(magnet_.size()),
      m0(magnet_.size()),
      m1(magnet_.size()) {
  // Necessary to make stepsize_ a vector? Sufficient to consider only smallest element?
  stepsize_ = {1e-14, 1e-14};
  for (int i = 0; i < magnet->sublattices().size(); i++) {
    torque_.push_back(relaxTorqueQuantity(magnet->sublattices()[i]));
  }
  // TODO: check if input arguments are sane
}

void Minimizer::exec() {
  nsteps_ = 0;
  lastMagDiffs_.clear();

  while (!converged())
    step();
}

__global__ void k_step(CuField mField,
                       const CuField m0Field,
                       const CuField torqueField,
                       real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!mField.cellInGrid(idx))
    return;

  real3 m0 = m0Field.vectorAt(idx);
  real3 t = torqueField.vectorAt(idx);

  real t2 = dt * dt * dot(t, t);
  real3 m = ((4 - t2) * m0 + 4 * dt * t) / (4 + t2);

  mField.setVectorInCell(idx, m);

}

static inline real BarzilianBorweinStepSize(Field& dm, Field& dtorque, int n) {
  real nom, div;
  if (n % 2 == 0) {
    nom = dotSum(dm, dm);
    div = dotSum(dm, dtorque);
  } else {
    nom = dotSum(dm, dtorque);
    div = dotSum(dtorque, dtorque);
  }
  if (div == 0.0)
    return 1e-14;  // TODO: figure out safe stepsize

  return nom / div;
}

void Minimizer::step() {
  for (int i = 0; i < magnet_.size(); i++) {

    m0[i] = magnet_[i]->magnetization()->eval();

    if (nsteps_ == 0)
      t0[i] = torque_[i].eval();
    else
      t0[i] = t1[i];

    m1[i] = Field(magnet_[i]->system(), 3);
    int N = m1[i].grid().ncells();
    cudaLaunch(N, k_step, m1[i].cu(), m0[i].cu(), t0[i].cu(), stepsize_[i]);
  }
  for (int i = 0; i < magnet_.size(); i++)
    magnet_[i]->magnetization()->set(m1[i]);  // normalizes
  for (int i = 0; i < magnet_.size(); i++)
    t1[i] = torque_[i].eval();

  for (int i = 0; i < magnet_.size(); i++) {
    Field dm = add(real(+1), m1[i], real(-1), m0[i]);
    Field dt = add(real(-1), t1[i], real(+1), t0[i]);  // TODO: check sign difference

    stepsize_[i] = BarzilianBorweinStepSize(dm, dt, nsteps_);

    addMagDiff(maxVecNorm(dm));
  }

  nsteps_ += 1;
}

bool Minimizer::converged() const {
  if (lastMagDiffs_.size() < nMagDiffSamples_)
    return false;
 
  for (auto dm : lastMagDiffs_)
    if (dm > stopMaxMagDiff_)
      return false;

  return true;
}

void Minimizer::addMagDiff(real dm) {
  lastMagDiffs_.push_back(dm);
  if (lastMagDiffs_.size() > nMagDiffSamples_)
    lastMagDiffs_.pop_front();
}
