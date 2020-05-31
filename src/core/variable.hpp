#pragma once

#include <string>

#include "fieldquantity.hpp"
#include "grid.hpp"

class Variable : public FieldQuantity {
 public:
  Variable(std::string name, std::string unit, int ncomp, Grid grid);
  ~Variable();

  int ncomp() const;
  Grid grid() const;
  std::string name() const;
  std::string unit() const;

  Field eval() const;

  const Field* field() const;

  virtual void set(const Field&) const;
  virtual void set(real) const;
  virtual void set(real3) const;

  // Assignment operators which call the respective set function
  void operator=(const Field& f) const { set(f); }
  void operator=(real val) const { set(val); }
  void operator=(real3 val) const { set(val); }

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
  void set(const Field&) const;
  void set(real) const;
  void set(real3) const;
};