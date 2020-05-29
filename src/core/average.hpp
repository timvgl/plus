#pragma once

#include "scalarquantity.hpp"

class FieldQuantity;

class Average : public ScalarQuantity {
 public:
  Average(FieldQuantity* parent, int component);

  real eval() const;
  std::string unit() const;
  std::string name() const;

 private:
  int comp_;
  FieldQuantity* parent_;
};