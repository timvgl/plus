#pragma once

#include "fieldquantity.hpp"
#include "grid.hpp"

enum MagnetFieldComputationMethod {
  MAGNETFIELDMETHOD_BRUTE,
  MAGNETFIELDMETHOD_FFT,
  MAGNETFIELDMETHOD_AUTO
};

class Parameter;
class Ferromagnet;
class Field;

class MagnetFieldExecutor {
 public:
  virtual void exec(Field* h, const Field* m, const Parameter* msat) const = 0;
};

/// MagnetField is a field quantity which evaluates the magnetic field of a
/// ferromagnet on a specified grid.
/// Note: if the specified grid matches the grid of the ferromagnet, then this
/// magnetic field is the demagnetization field of the magnet
class MagnetField : public FieldQuantity {
 public:
  MagnetField(Ferromagnet* magnet,
              Grid grid,
              MagnetFieldComputationMethod method = MAGNETFIELDMETHOD_AUTO);
  ~MagnetField();

  void setMethod(MagnetFieldComputationMethod);

  Ferromagnet * source() const;

  int ncomp() const;
  Grid grid() const;
  void evalIn(Field*) const;
  std::string unit() const;
  // TODO: std::string name() const;

  bool assuredZero() const;

 private:
  Ferromagnet* magnet_;
  Grid grid_;
  MagnetFieldExecutor* executor_;
};
