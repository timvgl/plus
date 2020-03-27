#pragma once

#include "quantity.hpp"

class Ferromagnet;

class FerromagnetQuantity : public Quantity {
 public:
  FerromagnetQuantity(Ferromagnet*,
                      int ncomp,
                      std::string name,
                      std::string unit);
  int ncomp() const;
  Grid grid() const;
  std::string name() const;
  std::string unit() const;

 protected:
  Ferromagnet* ferromagnet_;
  int ncomp_;
  std::string name_;
  std::string unit_;
};