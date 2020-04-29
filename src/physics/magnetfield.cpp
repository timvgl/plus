#include "magnetfield.hpp"

#include "ferromagnet.hpp"
#include "magnetfieldbrute.hpp"
#include "magnetfieldfft.hpp"
#include "parameter.hpp"
#include "world.hpp"

MagnetField::MagnetField(Ferromagnet* magnet,
                         Grid grid,
                         MagnetFieldComputationMethod method)
    : magnet_(magnet), grid_(grid) {
  switch (method) {
    case MAGNETFIELDMETHOD_AUTO:
      // TODO: make smart choice (dependent on the
      // grid sizes) when choosing between fft or
      // brute method. For now, we choose fft method
      executor_ = new MagnetFieldFFTExecutor(magnet->grid(),
                                             magnet->world()->cellsize());
      break;
    case MAGNETFIELDMETHOD_FFT:
      executor_ = new MagnetFieldFFTExecutor(magnet->grid(),
                                             magnet->world()->cellsize());
      break;
    case MAGNETFIELDMETHOD_BRUTE:
      executor_ = new MagnetFieldBruteExecutor(magnet->grid(),
                                               magnet->world()->cellsize());
      break;
  }
}

MagnetField::~MagnetField() {
  delete executor_;
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
  Parameter* msat = &magnet_->msat;
  const Field* m = magnet_->magnetization()->field();
  executor_->exec(h, m, msat);
}

std::string MagnetField::unit() const {
  return "T";
}

bool MagnetField::assuredZero() const {
  return magnet_->msat.assuredZero();
}