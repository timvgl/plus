#include "dynamicequation.hpp"

#include <exception>

#include "fieldquantity.hpp"
#include "variable.hpp"

DynamicEquation::DynamicEquation(const Variable* x,
                                 const FieldQuantity* rhs,
                                 const FieldQuantity* noiseTerm)
    : x(x), rhs(rhs), noiseTerm(noiseTerm) {
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

  if (noiseTerm) {
    if (x->grid() != noiseTerm->grid()) {
      throw std::runtime_error(
          "The variable and the noise term of a dynamic equation should have "
          "the same grid");
    }
    if (x->ncomp() != noiseTerm->ncomp()) {
      throw std::runtime_error(
          "The variable and the noise term of a dynamic equation should have "
          "the same number of components");
    }
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