#include "datatypes.hpp"
#include "field.hpp"
#include "parameter.hpp"

Parameter::Parameter(Grid grid, real value)
    : grid_(grid), field_(nullptr), uniformValue_(value) {}

Parameter::~Parameter() {
  if (field_)
    delete field_;
}

void Parameter::set(real value) {
  uniformValue_ = value;
  if (field_)
    delete field_;
}

void Parameter::set(Field* values) {
  if (!field_)
    field_ = new Field(grid_, 1);
  field_->copyFrom(values);
}

bool Parameter::isUniform() const {
  return !field_;
};

bool Parameter::isZero() const {
  return isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

Grid Parameter::grid() const {
  return grid_;
}

void Parameter::evalIn(Field* f) const {
  if (field_)
    f->copyFrom(field_);
  f->setUniformComponent(uniformValue_, 0);
}

CuParameter Parameter::cu() const {
  if (field_)
    return CuParameter{grid_, 0.0, field_->devptr(0)};
  return CuParameter{grid_, uniformValue_, nullptr};
}