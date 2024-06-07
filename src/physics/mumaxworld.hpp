#pragma once

#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "world.hpp"

class Antiferromagnet;
class Ferromagnet;
class Magnet;
class TimeSolver;

/** MumaxWorld is World with additional functionalities for the actual physics
 *  of mumax5.
 */
class MumaxWorld : public World {
 public:
  /** Construct a mumax world. */
  explicit MumaxWorld(real3 cellsize, Grid mastergrid = Grid(int3{0, 0, 0}));

  /** Destroy the world and all systems it contains. */
  ~MumaxWorld();

  /** Uniform bias magnetic field which will affect all magnets in the world. */
  real3 biasMagneticField;

  /** Add a magnet to the world. */
  Ferromagnet* addFerromagnet(Grid grid, std::string name = "");

  Antiferromagnet* addAntiferromagnet(Grid grid, std::string name = "");

  /** Add a magnet to the world with a non-trivial geometry. */
  Ferromagnet* addFerromagnet(Grid grid,
                              GpuBuffer<bool> geometry,
                              std::string name = "");

  Antiferromagnet* addAntiferromagnet(Grid grid,
                                      GpuBuffer<bool> geometry,
                                      std::string name = "");

  /** Get a magnet by its name.
   *  Return a nullptr if there is no magnet with specified name. */
  Magnet* getMagnet(std::string name) const;
  Ferromagnet* getFerromagnet(std::string name) const;
  Antiferromagnet* getAntiferromagnet(std::string name) const;


  /** Get map of all Magnets in this world. */
  const std::map<std::string, Magnet*> magnets() const;
  const std::map<std::string, Ferromagnet*> ferromagnets() const;
  const std::map<std::string, Antiferromagnet*> antiferromagnets() const;

 private:
  void resetTimeSolverEquations();

 private:
  std::map<std::string, std::unique_ptr<Magnet>> magnets_;
  std::map<std::string, std::unique_ptr<Ferromagnet>> ferromagnets_;
  std::map<std::string, std::unique_ptr<Antiferromagnet>> antiferromagnets_;
};
