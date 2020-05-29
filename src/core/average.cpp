#include "average.hpp"

#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "scalarquantity.hpp"
#include "reduce.hpp"

Average::Average(FieldQuantity* parent, int component)
    : parent_(parent), comp_(component) {}

real Average::eval() const {
    std::unique_ptr<Field> f = parent_->eval();
    return fieldComponentAverage(f.get());
}

std::string Average::unit() const { 
    return parent_->unit();
}

std::string Average::name() const { 
    std::string thename = parent_->name();
    if (parent_->ncomp()>1)
        thename += "_" + std::to_string(comp_);
    return thename;
}
