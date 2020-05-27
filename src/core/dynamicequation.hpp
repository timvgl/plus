#pragma once

class Variable;
class FieldQuantity;
class Grid;

class DynamicEquation {
 public:
  DynamicEquation(const Variable* x,
                  const FieldQuantity* rhs,
                  const FieldQuantity* noiseTerm = nullptr);

  const Variable* x;
  const FieldQuantity* rhs;
  const FieldQuantity* noiseTerm;

  int ncomp() const;
  Grid grid() const;
};
