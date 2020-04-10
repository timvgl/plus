#include "quantity.hpp"

#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
#include "reduce.hpp"

Quantity::~Quantity(){};

const Field* Quantity::cache() const {
  return nullptr;
}

std::string Quantity::unit() const {
  return "";
}

std::string Quantity::name() const {
  return "";
}

std::unique_ptr<Field> Quantity::eval() const {
  std::unique_ptr<Field> f(new Field(grid(), ncomp()));
  evalIn(f.get());
  return f;
}

std::vector<real> Quantity::average() const {
  std::unique_ptr<Field> f(new Field(grid(), ncomp()));
  evalIn(f.get());
  return fieldAverage(f.get());
}

bool Quantity::fieldCompatibilityCheck(const Field* f) const {
  return f->ncomp() == ncomp() && f->grid() == grid();
}