#pragma once

#include <functional>
#include <memory>
#include <string>

#include "antiferromagnet.hpp"
#include "fieldquantity.hpp"
#include "scalarquantity.hpp"
#include "system.hpp"

class Antiferromagnet;
class Ferromagnet;

typedef std::function<Field(const Antiferromagnet*)> AFM_FieldFunc;
typedef std::function<real(const Antiferromagnet*)> AFM_ScalarFunc;

class AFM_FieldQuantity : public FieldQuantity {
 public:
  AFM_FieldQuantity(const Antiferromagnet* magnet,
                    AFM_FieldFunc evalfunc,
                    int ncomp,
                    std::string name,
                    std::string unit)
      : magnet_(magnet),
        evalfunc_(evalfunc),
        ncomp_(ncomp),
        name_(name),
        unit_(unit) {}

  AFM_FieldQuantity* clone() {
    return new AFM_FieldQuantity(magnet_, evalfunc_, ncomp_, name_, unit_);
  }
  int ncomp() const { return ncomp_; }
  std::shared_ptr<const System> system() const {
    return magnet_->system();
  }
  std::string name() const { return name_; }
  std::string unit() const { return unit_; }

  Field eval() const { return evalfunc_(magnet_); }
  Field operator()() const { return evalfunc_(magnet_); }

 private:
  const Antiferromagnet* magnet_;
  int ncomp_;
  std::string name_;
  std::string unit_;
  AFM_FieldFunc evalfunc_;
};

class AFM_ScalarQuantity : public ScalarQuantity {
 public:
  AFM_ScalarQuantity(const Antiferromagnet* magnet,
                     AFM_ScalarFunc evalfunc,
                     std::string name,
                     std::string unit)
      : magnet_(magnet),
        evalfunc_(evalfunc),
        name_(name),
        unit_(unit) {}

  std::string name() const { return name_; }
  std::string unit() const { return unit_; }
  real eval() const { return evalfunc_(magnet_); }

 private:
  const Antiferromagnet* magnet_;
  std::string name_;
  std::string unit_;
  AFM_ScalarFunc evalfunc_;
};