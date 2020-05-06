#pragma once

#include <map>
#include <vector>

#include "datatypes.hpp"

class FieldQuantity;

class Table {
 public:
  Table();
  void addColumn(std::string name, FieldQuantity*, int comp);
  void writeLine();
  std::vector<real> getValues(std::string name) const;

 private:
  struct Column {
    FieldQuantity* quantity;
    int comp;
    std::vector<real> values;
    void writeValue();
  };
  std::map<std::string, Column> columns_;
  int nLines_;
};
