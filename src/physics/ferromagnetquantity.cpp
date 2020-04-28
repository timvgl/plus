#include "ferromagnetquantity.hpp"

#include "ferromagnet.hpp"

FerromagnetFieldQuantity::FerromagnetFieldQuantity(Ferromagnet* ferromagnet,
                                                   int ncomp,
                                                   std::string name,
                                                   std::string unit)
    : ferromagnet_(ferromagnet), ncomp_(ncomp), name_(name), unit_(unit) {}

int FerromagnetFieldQuantity::ncomp() const {
  return ncomp_;
}

Grid FerromagnetFieldQuantity::grid() const {
  return ferromagnet_->grid();
}

std::string FerromagnetFieldQuantity::name() const {
  return ferromagnet_->name() + ":" + name_;
}

std::string FerromagnetFieldQuantity::unit() const {
  return unit_;
}

FerromagnetScalarQuantity::FerromagnetScalarQuantity(Ferromagnet* ferromagnet,
                                                     std::string name,
                                                     std::string unit)
    : ferromagnet_(ferromagnet), name_(name), unit_(unit) {}

std::string FerromagnetScalarQuantity::name() const {
  return ferromagnet_->name() + ":" + name_;
}

std::string FerromagnetScalarQuantity::unit() const {
  return unit_;
}