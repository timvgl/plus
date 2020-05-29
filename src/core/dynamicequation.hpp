#pragma once

#include <memory>

class Variable;
class FieldQuantity;
class Grid;

class DynamicEquation {
 public:
  DynamicEquation(const Variable* x,
                  std::shared_ptr<FieldQuantity> rhs,
                  const FieldQuantity* noiseTerm = nullptr);

  const Variable* x;
  std::shared_ptr<FieldQuantity> rhs;
  const FieldQuantity* noiseTerm;

  int ncomp() const;
  Grid grid() const;
};
