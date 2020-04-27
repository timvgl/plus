#pragma once

#include "fieldquantity.hpp"

class Ferromagnet;

class FerromagnetFieldQuantity : public FieldQuantity {
 public:
  FerromagnetFieldQuantity(Ferromagnet*,
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