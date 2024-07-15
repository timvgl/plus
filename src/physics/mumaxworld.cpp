#include "mumaxworld.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "antiferromagnet.hpp"
#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "relaxer.hpp"
#include "system.hpp"
#include "thermalnoise.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

MumaxWorld::MumaxWorld(real3 cellsize, Grid mastergrid)
    : World(cellsize, mastergrid),
      biasMagneticField({0, 0, 0}),
      RelaxTorqueThreshold(-1.0) {}

MumaxWorld::~MumaxWorld() {}

void MumaxWorld::checkAddibility(Grid grid, std::string name) {
  if (!inMastergrid(grid)) {
      throw std::out_of_range(
          "Can not add magnet because the grid does not fit in the "
          "mastergrid ");
  }

  for (const auto& namedMagnet : magnets_) {
    Magnet* m = namedMagnet.second;
    if (grid.overlaps(m->grid())) {
      throw std::out_of_range(
          "Can not add magnet because it overlaps with another "
          "magnet.");
    }
  }

  if (magnets_.find(name) != magnets_.end()) {
    throw std::runtime_error("A magnet with the name '" + name +
                             "' already exists");
  }
}

Ferromagnet* MumaxWorld::addFerromagnet(Grid grid, std::string name) {
  return addFerromagnet(grid, GpuBuffer<bool>(), name);
}

Antiferromagnet* MumaxWorld::addAntiferromagnet(Grid grid, std::string name) {
  return addAntiferromagnet(grid, GpuBuffer<bool>(), name);
}

Ferromagnet* MumaxWorld::addFerromagnet(Grid grid,
                                        GpuBuffer<bool> geometry,
                                        std::string name) {
  // Create name if not given.
  static int idxUnnamed = 1;
  if (name.length() == 0) {
    name = "ferromagnet_" + std::to_string(idxUnnamed++);
  }

  // Check if Ferromagnet can be added to this world.
  checkAddibility(grid, name);

  // Create the magnet and add it to this world
  ferromagnets_[name] =
      std::make_unique<Ferromagnet>(this, grid, name, geometry);

  Ferromagnet* newMagnet = ferromagnets_[name].get();
  magnets_[name] = newMagnet;
  
  handleNewStrayfield(newMagnet);
  resetTimeSolverEquations();
  return newMagnet;
}

Antiferromagnet* MumaxWorld::addAntiferromagnet(Grid grid,
                                                GpuBuffer<bool> geometry,
                                                std::string name) {
  // Create name if not given.
  static int idxUnnamed = 1;
  if (name.length() == 0) {
    name = "antiferromagnet_" + std::to_string(idxUnnamed++);
  }                

  // Check if Antiferromagnet can be added to this world.
  checkAddibility(grid, name);

  // Create the magnet and add it to this world
  antiferromagnets_[name] =
      std::make_unique<Antiferromagnet>(this, grid, name, geometry);
  Antiferromagnet* newMagnet = antiferromagnets_[name].get();
  magnets_[name] = newMagnet;

  handleNewStrayfield(newMagnet);
  resetTimeSolverEquations();
  return newMagnet;
}

void MumaxWorld::handleNewStrayfield(Magnet* newMagnet) {
  for (const auto& namedMagnet : magnets_) {
    Magnet* otherMagnet = namedMagnet.second;
    if (otherMagnet != nullptr) {
      otherMagnet->addStrayField(newMagnet);
      // Avoid adding the field on itself twice
      if (otherMagnet != newMagnet)
        newMagnet->addStrayField(otherMagnet);
    }
  }
}

Magnet* MumaxWorld::getMagnet(std::string name) const {
  auto namedMagnet = magnets_.find(name);
  if (namedMagnet == magnets_.end())
    return nullptr;
  return namedMagnet->second;
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
  return magnets_;
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

void MumaxWorld::resetTimeSolverEquations(FM_Field torque) const {
  std::vector<DynamicEquation> equations;
  for (const auto& namedMagnet : ferromagnets_) {
    Ferromagnet* magnet = namedMagnet.second.get();
    DynamicEquation eq(
        magnet->magnetization(),
        std::shared_ptr<FieldQuantity>(torque(magnet).clone()),
        std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(magnet).clone()));
    equations.push_back(eq);
  }

  for (const auto& namedMagnet : antiferromagnets_) {
    const Antiferromagnet* magnet = namedMagnet.second.get();
    for (const Ferromagnet* sub : magnet->sublattices()) {
      DynamicEquation eq(
        sub->magnetization(),
        std::shared_ptr<FieldQuantity>(torque(sub).clone()),
        std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(sub).clone()));
      equations.push_back(eq);
    }
  }
  timesolver_->setEquations(equations);
}

void MumaxWorld::relax(real tol) {
    Relaxer relaxer(this, this->RelaxTorqueThreshold, tol);
    relaxer.exec();
}