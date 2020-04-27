#pragma once

#include <string>

#include "grid.hpp"
#include "quantity.hpp"

class Variable : public Quantity {
 public:
  Variable(std::string name, std::string unit, int ncomp, Grid grid);
  ~Variable();

  int ncomp() const;
  Grid grid() const;
  std::string name() const;
  std::string unit() const;

  void evalIn(Field*) const;

  const Field* field() const;

  virtual void set(const Field*) const;
  virtual void set(real) const;
  virtual void set(real3) const;

 protected:
  Field* field_;

 private:
  std::string name_;
  std::string unit_;
};

// Exactly the same as variable, but when values are set, the values are
// normalized
class NormalizedVariable : public Variable {
 public:
  NormalizedVariable(std::string name, std::string unit, int ncomp, Grid grid);
  void set(const Field*) const;
  void set(real) const;
  void set(real3) const;
};