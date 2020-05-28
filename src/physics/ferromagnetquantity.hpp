#pragma once

#include <functional>

#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "handler.hpp"
#include "scalarquantity.hpp"

class Ferromagnet;

class FerromagnetFieldQuantity : public FieldQuantity {
 public:
  FerromagnetFieldQuantity(Handle<Ferromagnet>,
                           int ncomp,
                           std::string name,
                           std::string unit);
  int ncomp() const;
  Grid grid() const;
  std::string name() const;
  std::string unit() const;

 protected:
  Handle<Ferromagnet> ferromagnet_;
  int ncomp_;
  std::string name_;
  std::string unit_;
};

class FerromagnetScalarQuantity : public ScalarQuantity {
 public:
  FerromagnetScalarQuantity(Handle<Ferromagnet>,
                            std::string name,
                            std::string unit);
  std::string name() const;
  std::string unit() const;

 protected:
  Handle<Ferromagnet> ferromagnet_;
  std::string name_;
  std::string unit_;
};