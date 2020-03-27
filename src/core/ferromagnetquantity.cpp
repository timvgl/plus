#include "ferromagnetquantity.hpp"

#include "ferromagnet.hpp"

FerromagnetQuantity::FerromagnetQuantity(Ferromagnet* ferromagnet,
                                         int ncomp,
                                         std::string name,
                                         std::string unit)
    : ferromagnet_(ferromagnet), ncomp_(ncomp), name_(name), unit_(unit) {}

int FerromagnetQuantity::ncomp() const {
  return ncomp_;
}

Grid FerromagnetQuantity::grid() const {
  return ferromagnet_->grid();
}

std::string FerromagnetQuantity::name() const {
  return ferromagnet_->name() + ":" + name_;
}

std::string FerromagnetQuantity::unit() const {
  return unit_;
}