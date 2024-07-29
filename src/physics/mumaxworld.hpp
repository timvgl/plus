#pragma once

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "datatypes.hpp"
#include "ferromagnetquantity.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "torque.hpp"
#include "world.hpp"

class Antiferromagnet;
class Ferromagnet;
class FM_FieldQuantity;
class Magnet;
class TimeSolver;

typedef std::function<FM_FieldQuantity(const Ferromagnet*)> FM_Field;

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

  void checkAddibility(Grid grid, std::string name);

  /** Add a ferromagnet to the world. */
  Ferromagnet* addFerromagnet(Grid grid, std::string name = "");
  /** Add an antiferromagnet to the world. */
  Antiferromagnet* addAntiferromagnet(Grid grid, std::string name = "");

  /** Add a ferromagnet to the world with a non-trivial geometry. */
  Ferromagnet* addFerromagnet(Grid grid,
                              GpuBuffer<bool> geometry,
                              std::string name = "");
  /** Add an antiferromagnet to the world with a non-trivial geometry. */
  Antiferromagnet* addAntiferromagnet(Grid grid,
                                      GpuBuffer<bool> geometry,
                                      std::string name = "");

    
  /**Add the magnetic field of the other magnets in the new magnet, and vice versa. */
  void handleNewStrayfield(Magnet* newMagnet);

  /** Get a magnet by its name.
   *  Return a nullptr if there is no magnet with specified name. */
  Magnet* getMagnet(std::string name) const;
  /** Get a ferromagnet by its name.
   *  Return a nullptr if there is no ferromagnet with specified name. */
  Ferromagnet* getFerromagnet(std::string name) const;
  /** Get an antiferromagnet by its name.
   *  Return a nullptr if there is no antiferromagnet with specified name. */
  Antiferromagnet* getAntiferromagnet(std::string name) const;


  /** Get map of all Magnets in this world. */
  const std::map<std::string, Magnet*> magnets() const;
  /** Get map of all Ferromagnets in this world. */
  const std::map<std::string, Ferromagnet*> ferromagnets() const;
  /** Get map of all Antiferromagnets in this world. */
  const std::map<std::string, Antiferromagnet*> antiferromagnets() const;

  /** Minimize the current energy state of the world with every magnet in it. */
  void minimize(real tol = 1e-6, int nSamples = 10);
  /** Relax the current state of the world with every magnet in it. */
  void relax(real tol);
  real RelaxTorqueThreshold;

  void resetTimeSolverEquations(FM_Field torque = torqueQuantity) const;

 private:
  std::map<std::string, Magnet*> magnets_;
  std::map<std::string, std::unique_ptr<Ferromagnet>> ferromagnets_;
  std::map<std::string, std::unique_ptr<Antiferromagnet>> antiferromagnets_;
};
