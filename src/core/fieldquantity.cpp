#include "fieldquantity.hpp"

#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
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

std::vector<real> FieldQuantity::average() const {
  std::unique_ptr<Field> f(new Field(grid(), ncomp()));
  evalIn(f.get());
  return fieldAverage(f.get());
}

bool FieldQuantity::fieldCompatibilityCheck(const Field* f) const {
  return f->ncomp() == ncomp() && f->grid() == grid();
}