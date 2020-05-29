#include "table.hpp"

#include <map>
#include <memory>
#include <vector>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"

Table::Table() : nLines_(0) {}

void Table::addColumn(std::string name, FieldQuantity* q, int comp) {
  if (columns_.find(name) != columns_.end()) {
    throw std::runtime_error(
        "Can not add a column with the name " + name +
        " because there exists already a column with that name.");
  }
  real nan = 0.0 / 0.0;
  columns_.insert({name, Column{q, comp, std::vector<real>(nLines_, nan)}});
}

void Table::writeLine() {
  for (auto& column : columns_) {
    column.second.writeValue();
  }
}

std::vector<real> Table::getValues(std::string name) const {
  if (columns_.find(name) == columns_.end()) {
    throw std::runtime_error("Did not find a column with the name " + name);
  }
  return columns_.find(name)->second.values;
}

void Table::Column::writeValue() {
  Field field = quantity->eval();
  real value = fieldComponentAverage(&field, comp);
  values.push_back(value);
}