#include "fieldquantity.hpp"

#include <stdexcept>
#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"

FieldQuantity::~FieldQuantity(){};

const Field* FieldQuantity::cache() const {
  return nullptr;
}

std::string FieldQuantity::unit() const {
  return "";
}

std::string FieldQuantity::name() const {
  return "";
}

void FieldQuantity::addTo(Field* f) const {
  if (!fieldCompatibilityCheck(f))
    throw std::invalid_argument(
        "Can not add the quantity to given field because the fields are "
        "incompatible.");
  if (assuredZero())
    return;
  Field result = eval();
  add(f, f, &result);
}

std::vector<real> FieldQuantity::average() const {
  Field f = eval();
  return fieldAverage(&f);
}

bool FieldQuantity::assuredZero() const {
  return false;
}

bool FieldQuantity::fieldCompatibilityCheck(const Field* f) const {
  return f->ncomp() == ncomp() && f->grid() == grid();
}