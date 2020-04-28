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

std::unique_ptr<Field> FieldQuantity::eval() const {
  std::unique_ptr<Field> f(new Field(grid(), ncomp()));
  evalIn(f.get());
  return f;
}

void FieldQuantity::addTo(Field* f) const {
  if (!fieldCompatibilityCheck(f))
    throw std::invalid_argument(
        "Can not add the quantity to given field because the fields are "
        "incompatible.");
  if (assuredZero())
    return;
  auto result = eval();
  add(f, f, result.get());
}

std::vector<real> FieldQuantity::average() const {
  std::unique_ptr<Field> f(new Field(grid(), ncomp()));
  evalIn(f.get());
  return fieldAverage(f.get());
}

bool FieldQuantity::assuredZero() const {
  return false;
}

bool FieldQuantity::fieldCompatibilityCheck(const Field* f) const {
  return f->ncomp() == ncomp() && f->grid() == grid();
}