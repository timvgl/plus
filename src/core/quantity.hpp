#pragma once

#include <memory>
#include <string>
#include <vector>

#include "datatypes.hpp"

class Grid;
class Field;

/// Quantity interface
class Quantity {
 public:
  /// Virtual destructor which does nothing
  virtual ~Quantity();

  /***** PURE VIRTUAL FUNCTIONS *****/

  /// Returns the number of components of the quantity. In most cases this
  /// would be either 1 (scalar field) or 3 (vector fields)
  virtual int ncomp() const = 0;

  /// Returns the grid on which the quantity lives
  virtual Grid grid() const = 0;

  /// Evaluates the quantity and saves the results on a given field object. The
  /// field object should live on the same grid as the quantity and should have
  /// the same number of components
  virtual void evalIn(Field*) const = 0;

  /***** NON-PURE VIRTUAL FUNCTIONS *****/

  /// Quantity values which do not need to be computed (e.g. variables,
  /// parameters, or cached quantity values) can be accessed via a pointer to
  /// the field object in which the values are stored.
  virtual const Field* cache() const;

  /// Returns the unit of the quantity (empty string by default)
  virtual std::string unit() const;

  /// Returns the name of the quantity (empty string by default)
  virtual std::string name() const;

  /// Allocate memory for a field and calls void eval(Field *)
  virtual std::unique_ptr<Field> eval() const;

  /// Eval the quantity and return the average of each component
  virtual std::vector<real> average() const;

  /***** NON-VIRTUAL HELPER FUNCTIONS *****/

  /// Checks if the quantity can be evaluated in a given field object
  bool fieldCompatibilityCheck(const Field*) const;
};