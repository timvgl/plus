#include "dynamicequation.hpp"

#include <exception>

#include "quantity.hpp"
#include "variable.hpp"

DynamicEquation::DynamicEquation(Variable* x, Quantity* rhs) : x(x), rhs(rhs) {
  if (x->grid() != rhs->grid()) {
    throw std::runtime_error(
        "The variable and the r.h.s. of a dynamic equation should have the "
        "same grid");
  }
  if (x->ncomp() != rhs->ncomp()) {
    throw std::runtime_error(
        "The variable and the r.h.s. of a dynamic equation should have the "
        "same number of components");
  }
}

int DynamicEquation::ncomp() const {
  return x->ncomp();
  ;
}

Grid DynamicEquation::grid() const {
  return x->grid();
  ;
}