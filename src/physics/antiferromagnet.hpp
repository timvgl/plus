#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "world.hpp"

class Antiferromagnet : public Magnet {
 public:
  Antiferromagnet(MumaxWorld* world,
        Grid grid,
         std::string name,
         GpuBuffer<bool> geometry);
  ~Antiferromagnet() override = default;
  
 std::vector<Ferromagnet*> sublattices() const;
 Ferromagnet* sub1();
 Ferromagnet* sub2();

 public:
  Parameter afmex_cell;
  Parameter afmex_nn;
  Parameter latcon;
  Ferromagnet sub1_;
  Ferromagnet sub2_;
 
 private:
  std::vector<Ferromagnet*> sublattices_ = {&sub1_, &sub2_};
};