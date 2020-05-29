#include "magnetfield.hpp"

#include "ferromagnet.hpp"
#include "magnetfieldbrute.hpp"
#include "magnetfieldfft.hpp"
#include "parameter.hpp"
#include "world.hpp"

MagnetField::MagnetField(const Ferromagnet * magnet,
                         Grid grid,
                         MagnetFieldComputationMethod method)
    : magnet_(magnet), grid_(grid), executor_(nullptr) {
  setMethod(method);
}

MagnetField::~MagnetField() {
  if (executor_)
    delete executor_;
}

void MagnetField::setMethod(MagnetFieldComputationMethod method) {
  // TODO: check if method has been changed. If not, do nothing
  if (executor_)
    delete executor_;

  switch (method) {
    case MAGNETFIELDMETHOD_AUTO:
      // TODO: make smart choice (dependent on the
      // grid sizes) when choosing between fft or
      // brute method. For now, we choose fft method
      executor_ = new MagnetFieldFFTExecutor(grid_, magnet_->grid(),
                                             magnet_->world()->cellsize());
      break;
    case MAGNETFIELDMETHOD_FFT:
      executor_ = new MagnetFieldFFTExecutor(grid_, magnet_->grid(),
                                             magnet_->world()->cellsize());
      break;
    case MAGNETFIELDMETHOD_BRUTE:
      executor_ = new MagnetFieldBruteExecutor(grid_, magnet_->grid(),
                                               magnet_->world()->cellsize());
      break;
  }
}

const Ferromagnet * MagnetField::source() const {
  return magnet_;
}

int MagnetField::ncomp() const {
  return 3;
}

Grid MagnetField::grid() const {
  return grid_;
}

void MagnetField::evalIn(Field* h) const {
  if (assuredZero()) {
    h->makeZero();
    return;
  }
  const Parameter* msat = &magnet_->msat;
  const Field* m = magnet_->magnetization()->field();
  executor_->exec(h, m, msat);
}

Field MagnetField::eval() const {
  Field h(grid(), ncomp());
  if (assuredZero()) {
    h.makeZero();
    return h;
  }
  const Parameter* msat = &magnet_->msat;
  const Field* m = magnet_->magnetization()->field();
  executor_->exec(&h, m, msat);
  return h;
}

std::string MagnetField::unit() const {
  return "T";
}

bool MagnetField::assuredZero() const {
  return magnet_->msat.assuredZero();
}