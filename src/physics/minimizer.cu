#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "minimizer.hpp"
#include "reduce.hpp"
#include "torque.hpp"

Minimizer::Minimizer(Handle<Ferromagnet> magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : magnet_(magnet),
      torque_(magnet_),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff) {
  stepsize_ = 1e-14;  // TODO: figure out how to make descent guess

  // TODO: check if input arguments are sane

  t_new = new Field(magnet_->grid(), 3);
  t_old = new Field(magnet_->grid(), 3);
  m_new = new Field(magnet_->grid(), 3);
  m_old = new Field(magnet_->grid(), 3);
}

Minimizer::~Minimizer() {
  delete t_new, t_old, m_new, m_old;
}

void Minimizer::exec() {
  nsteps_ = 0;
  lastMagDiffs_.clear();

  while (!converged())
    step();
}

__global__ static void k_step(CuField mField,
                              CuField m0Field,
                              CuField torqueField,
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

static inline real BarzilianBorweinStepSize(Field* dm, Field* dtorque, int n) {
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
  m_old->copyFrom(magnet_->magnetization()->field());

  if (nsteps_ == 0)
    torque_.evalIn(t_old);
  else
    t_old->copyFrom(t_new);

  int N = m_new->grid().ncells();
  cudaLaunch(N, k_step, m_new->cu(), m_old->cu(), t_old->cu(), stepsize_);

  magnet_->magnetization()->set(m_new);  // normalizes

  torque_.evalIn(t_new);

  auto dm = m_old;  // let's reuse m_old
  auto dt = t_old;  // "
  add(dm, +1, m_new, -1, m_old);
  add(dt, -1, t_new, +1, t_old);  // TODO: check sign difference

  stepsize_ = BarzilianBorweinStepSize(dm, dt, nsteps_);

  addMagDiff(maxVecNorm(dm));

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
