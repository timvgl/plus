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
  if (field_) {
    delete field_;
    field_ = nullptr;
  }
}

void Parameter::set(const Field& values) {
  if (!field_)
    field_ = new Field(grid_, 1);
  field_->copyFrom(values);
}

bool Parameter::isUniform() const {
  return !field_;
};

bool Parameter::assuredZero() const {
  return isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

Grid Parameter::grid() const {
  return grid_;
}

Field Parameter::eval() const {
  Field p(grid(), ncomp());
  if (field_) {
    p.copyFrom(*field_);
  } else {
    p.setUniformComponent(uniformValue_, 0);
  }
  return p;
}

CuParameter Parameter::cu() const {
  if (isUniform())
    return CuParameter(grid_, uniformValue_);
  return CuParameter(grid_, field_->devptr(0));
}

VectorParameter::VectorParameter(Grid grid, real3 value)
    : grid_(grid), field_(nullptr), uniformValue_(value) {}

VectorParameter::~VectorParameter() {
  if (field_)
    delete field_;
}

void VectorParameter::set(real3 value) {
  uniformValue_ = value;
  if (field_)
    delete field_;
}

void VectorParameter::set(const Field& values) {
  if (!field_)
    field_ = new Field(grid_, 3);
  field_->copyFrom(values);
}

bool VectorParameter::isUniform() const {
  return !field_;
};

bool VectorParameter::assuredZero() const {
  return isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

Grid VectorParameter::grid() const {
  return grid_;
}

Field VectorParameter::eval() const {
  Field p(grid(), ncomp());
  if (field_)
    p.copyFrom(*field_);
  p.setUniformComponent(uniformValue_.x, 0);
  p.setUniformComponent(uniformValue_.y, 1);
  p.setUniformComponent(uniformValue_.z, 2);
  return p;
}

CuVectorParameter VectorParameter::cu() const {
  if (isUniform())
    return CuVectorParameter(grid_, uniformValue_);
  return CuVectorParameter{grid_, field_->devptr(0), field_->devptr(1),
                           field_->devptr(2)};
}