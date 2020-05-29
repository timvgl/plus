#include "world.hpp"

#include <handler.hpp>
#include <stdexcept>
#include <string>

#include "ferromagnet.hpp"

World::World(real3 cellsize)
    : cellsize_(cellsize), biasMagneticField({0, 0, 0}) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }
};

World::~World() {
  for (auto fm : Ferromagnets) {
    delete fm.second;
  }
}

real3 World::cellsize() const {
  return cellsize_;
}

real World::cellVolume() const {
  return cellsize_.x * cellsize_.y * cellsize_.z;
}

Ferromagnet* World::addFerromagnet(Grid grid, std::string name) {
  for (const auto& fm : Ferromagnets)
    if (grid.overlaps(fm.second->grid()))
      throw std::out_of_range(
          "Can not add ferromagnet because it overlaps with another "
          "ferromagnet.");

  static int idxUnnamed = 1;
  if (name.length() == 0)
    name = "magnet_" + std::to_string(idxUnnamed++);

  if (Ferromagnets.find(name) != Ferromagnets.end())
    throw std::runtime_error("A ferromagnet with the name '" + name +
                             "' already exists");
  Ferromagnet* newMagnet = new Ferromagnet(this, name, grid);
  Ferromagnets[name] = newMagnet;

  // Add the magnetic field of the other magnets in this magnet, and vice versa
  for (auto entry : Ferromagnets) {
    Ferromagnet* magnet = entry.second;
    magnet->addMagnetField(newMagnet, MAGNETFIELDMETHOD_AUTO);
    if (magnet != newMagnet) {  // Avoid adding the field on itself twice
      newMagnet->addMagnetField(magnet, MAGNETFIELDMETHOD_AUTO);
    }
  }

  return newMagnet;
};

Ferromagnet* World::getFerromagnet(std::string name) const {
  auto result = Ferromagnets.find(name);
  if (result == Ferromagnets.end())
    return nullptr;
  return result->second;
}