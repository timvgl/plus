#pragma once

#include <curand.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dmitensor.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "poissonsystem.hpp"
#include "strayfield.hpp"
#include "variable.hpp"
#include "world.hpp"

class FieldQuantity;
class MumaxWorld;
class System;

class Magnet {
 public:
  Magnet(MumaxWorld* world,
         Grid grid,
         std::string name,
         GpuBuffer<bool> geometry = GpuBuffer<bool>());
  Magnet(Magnet&&) = default;  // TODO: check if default is ok

  std::string name() const;
  std::shared_ptr<const System> system() const;
  const World* world() const;
  Grid grid() const;
  real3 cellsize() const;
  const GpuBuffer<bool>& getGeometry() const;


 public:
  std::shared_ptr<System> system_;  // the system_ has to be initialized first,
                                    // hence its listed as the first datamember here
  std::string name_;
  int ncomp_;

 private:
  Magnet(const Magnet&);
  Magnet& operator=(const Magnet&);

/* TO DO - global properties / parameters / ...
* Poissonsystem?
* STT-parameters?
*/
};
