#include "variable.hpp"

#include "field.hpp"
#include "fieldops.hpp"

Variable::Variable(std::string name, std::string unit, int ncomp, Grid grid)
    : name_(name), unit_(unit) {
  field_ = new Field(grid, ncomp);
}

Variable::~Variable() {
  delete field_;
}

int Variable::ncomp() const {
  return field_->ncomp();
}

Grid Variable::grid() const {
  return field_->grid();
}

std::string Variable::name() const {
  return name_;
}

std::string Variable::unit() const {
  return unit_;
}

void Variable::evalIn(Field* result) const {
  result->copyFrom(field_);
}

const Field* Variable::field() const {
  return field_;
}

void Variable::set(const Field* src) const {
  field_->copyFrom(src);
}

NormalizedVariable::NormalizedVariable(std::string name,
                                       std::string unit,
                                       int ncomp,
                                       Grid grid)
    : Variable(name, unit, ncomp, grid) {}

void NormalizedVariable::set(const Field* src) const {
  // TODO: check if this is possible without the extra copy
  auto f = normalized(src);
  field_->copyFrom(f.get());
}