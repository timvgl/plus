#pragma once

#include <memory>

#include "field.hpp"

// add two fields
void add(Field* y, real a1, const Field* x1, real a2, const Field* x2);
void add(Field* y, const Field* x1, const Field* x2);
void add(Field* y, std::vector<const Field*> x, std::vector<real> weights);

void addConstant(Field* y, Field* x, real value, int comp = 0);

void normalized(Field* dst, const Field* src);
std::unique_ptr<Field> normalized(const Field* src);
void normalize(Field*);
