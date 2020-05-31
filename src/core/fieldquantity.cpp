#include "fieldquantity.hpp"

#include <stdexcept>
#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"

void FieldQuantity::addTo(Field* f) const {
  if (!fieldCompatibilityCheck(f))
    throw std::invalid_argument(
        "Can not add the quantity to given field because the fields are "
        "incompatible.");
  if (assuredZero())
    return;
  (*f) += eval();
}

std::vector<real> FieldQuantity::average() const {
  return fieldAverage(eval());
}

bool FieldQuantity::fieldCompatibilityCheck(const Field* f) const {
  return f->ncomp() == ncomp() && f->grid() == grid();
}