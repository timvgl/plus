#pragma once

class Variable;
class FieldQuantity;
class Grid;

class DynamicEquation {
 public:
  DynamicEquation(const Variable *x, const FieldQuantity* rhs);
  const Variable* x;
  const FieldQuantity* rhs;

  int ncomp() const;
  Grid grid() const;
};