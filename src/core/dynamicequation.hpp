#pragma once

class Variable;
class Quantity;
class Grid;

class DynamicEquation {
 public:
  DynamicEquation(Variable *x, Quantity* rhs);
  const Variable* x;
  const Quantity* rhs;

  int ncomp() const;
  Grid grid() const;
};