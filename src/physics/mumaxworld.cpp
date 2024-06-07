#include "mumaxworld.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include "antiferromagnet.hpp"
#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "system.hpp"
#include "thermalnoise.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

MumaxWorld::MumaxWorld(real3 cellsize, Grid mastergrid)
    : World(cellsize, mastergrid), biasMagneticField({0, 0, 0}) {}

MumaxWorld::~MumaxWorld() {}

Ferromagnet* MumaxWorld::addFerromagnet(Grid grid, std::string name) {
  return addFerromagnet(grid, GpuBuffer<bool>(), name);
}

Antiferromagnet* MumaxWorld::addAntiferromagnet(Grid grid, std::string name) {
  return addAntiferromagnet(grid, GpuBuffer<bool>(), name);
}

Ferromagnet* MumaxWorld::addFerromagnet(Grid grid,
                                        GpuBuffer<bool> geometry,
                                        std::string name) {
  if (!inMastergrid(grid)) {
    throw std::out_of_range(
        "Can not add ferromagnet because the grid does not fit in the "
        "mastergrid ");
  }

  for (const auto& namedMagnet : magnets_) {
    Magnet* m = namedMagnet.second.get();
    if (grid.overlaps(m->grid())) {
      throw std::out_of_range(
          "Can not add ferromagnet because it overlaps with another "
          "magnet.");
    }
  }

  static int idxUnnamed = 1;
  if (name.length() == 0) {
    name = "ferromagnet_" + std::to_string(idxUnnamed++);
  }

  if (ferromagnets_.find(name) != ferromagnets_.end()) {
    throw std::runtime_error("A ferromagnet with the name '" + name +
                             "' already exists");
  }
  
  // Create the magnet and add it to this world
  ferromagnets_[name] =
      std::make_unique<Ferromagnet>(this, grid, name, geometry);
  magnets_[name] = 
      std::make_unique<Ferromagnet>(this, grid, name, geometry);
  Ferromagnet* newMagnet = ferromagnets_[name].get();
  
  // Add the magnetic field of the other magnets in this magnet, and vice versa
  for (const auto& namedMagnet : ferromagnets_) {
    Ferromagnet* otherMagnet = namedMagnet.second.get();
    if (otherMagnet != nullptr) {
      otherMagnet->addStrayField(newMagnet);
      // Avoid adding the field on itself twice
      if (otherMagnet != newMagnet) {
        newMagnet->addStrayField(otherMagnet);
      }
    }
  }
  resetTimeSolverEquations();
  return newMagnet;
}

Antiferromagnet* MumaxWorld::addAntiferromagnet(Grid grid,
                                                GpuBuffer<bool> geometry,
                                                std::string name) {
  if (!inMastergrid(grid)) {
    throw std::out_of_range(
        "Can not add antiferromagnet because the grid does not fit in the "
        "mastergrid ");
  }

  for (const auto& namedMagnet : magnets_) {
    Magnet* m = namedMagnet.second.get();
    if (grid.overlaps(m->grid())) {
      throw std::out_of_range(
          "Can not add antiferromagnet because it overlaps with another "
          "magnet.");
    }
  }

  static int idxUnnamed = 1;
  if (name.length() == 0) {
    name = "antiferromagnet_" + std::to_string(idxUnnamed++);
  }

  if (antiferromagnets_.find(name) != antiferromagnets_.end()) {
    throw std::runtime_error("An antiferromagnet with the name '" + name +
                             "' already exists");
  }
  
  // Create the magnet and add it to this world
  antiferromagnets_[name] =
      std::make_unique<Antiferromagnet>(this, grid, name, geometry);
  magnets_[name] = 
      std::make_unique<Antiferromagnet>(this, grid, name, geometry);
  Antiferromagnet* newMagnet = antiferromagnets_[name].get();
  
  /* TO DO:
  Add the magnetic field of the other magnets in this AFM, and vice versa
  */

  resetTimeSolverEquations();
  return newMagnet;
}

Magnet* MumaxWorld::getMagnet(std::string name) const {
  auto namedMagnet = magnets_.find(name);
  if (namedMagnet == magnets_.end())
    return nullptr;
  return namedMagnet->second.get();
}

Ferromagnet* MumaxWorld::getFerromagnet(std::string name) const {
  auto namedMagnet = ferromagnets_.find(name);
  if (namedMagnet == ferromagnets_.end())
    return nullptr;
  return namedMagnet->second.get();
}

Antiferromagnet* MumaxWorld::getAntiferromagnet(std::string name) const {
  auto namedMagnet = antiferromagnets_.find(name);
  if (namedMagnet == antiferromagnets_.end())
    return nullptr;
  return namedMagnet->second.get();
}

const std::map<std::string, Magnet*> MumaxWorld::magnets() const {
  std::map<std::string, Magnet*> sharedMagnets;
  for (const auto& pair : magnets_) {
    sharedMagnets[pair.first] = pair.second.get();
  }
  return sharedMagnets;
}

const std::map<std::string, Ferromagnet*> MumaxWorld::ferromagnets() const {
  std::map<std::string, Ferromagnet*> sharedFerromagnets;
  for (const auto& pair : ferromagnets_) {
    sharedFerromagnets[pair.first] = pair.second.get();
  }
  return sharedFerromagnets;
}

const std::map<std::string, Antiferromagnet*> MumaxWorld::antiferromagnets() const {
  std::map<std::string, Antiferromagnet*> sharedAntiferromagnets;
  for (const auto& pair : antiferromagnets_) {
    sharedAntiferromagnets[pair.first] = pair.second.get();
  }
  return sharedAntiferromagnets;
}

void MumaxWorld::resetTimeSolverEquations() {
  std::vector<DynamicEquation> equations;
  for (const auto& namedMagnet : ferromagnets_) {
    std::cout << "add ferro";
    Ferromagnet* magnet = namedMagnet.second.get();
    DynamicEquation eq(
        magnet->magnetization(),
        std::shared_ptr<FieldQuantity>(torqueQuantity(magnet).clone()),
        std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(magnet).clone()));
    equations.push_back(eq);
  }

  for (const auto& namedMagnet : antiferromagnets_) {
    std::cout << "add anti";
    Antiferromagnet* magnet = namedMagnet.second.get();
    for (Ferromagnet* sub : magnet->sublattices()) {
      DynamicEquation eq(
        sub->magnetization(),
        std::shared_ptr<FieldQuantity>(torqueQuantity(sub).clone()),
        std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(sub).clone()));
      equations.push_back(eq);
    }
  }
  timesolver_->setEquations(equations);
}
