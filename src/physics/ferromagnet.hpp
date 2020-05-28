#pragma once

#include <map>
#include <string>
#include <vector>
#include "field.hpp"
#include "grid.hpp"
#include "magnetfield.hpp"
#include "parameter.hpp"
#include "system.hpp"
#include "variable.hpp"
#include "handler.hpp"

class FieldQuantity;
class World;

class Ferromagnet : public System {
 public:
  Ferromagnet(World* world, std::string name, Grid grid);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;  // TODO: check if default is ok

  // A ferromagnet should have a handler in the world.
  // This function obtains a handle from this handler
  // TODO: check if this is ok, feels hacky
  Handle<Ferromagnet> getHandle() const;  

  const Variable* magnetization() const;

  bool enableDemag;

  VectorParameter anisU;
  Parameter msat;
  Parameter aex;
  Parameter ku1;
  Parameter alpha;
  Parameter temperature;

  const MagnetField* getMagnetField(Handle<Ferromagnet>) const;
  std::vector<const MagnetField*> getMagnetFields() const;
  void addMagnetField(
      Handle<Ferromagnet>,
      MagnetFieldComputationMethod method = MAGNETFIELDMETHOD_BRUTE);
  void removeMagnetField(Handle<Ferromagnet>);

  void minimize(real tol = 1e-6, int nSamples = 10);

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  NormalizedVariable magnetization_;
  std::map<Handle<Ferromagnet>, MagnetField*> magnetFields_;
};