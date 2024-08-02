#pragma once

#include <memory>

#include "datatypes.hpp"
#include "grid.hpp"

class TimeSolver;

class World {
 public:
  /** Create a world with a given cell size and master grid.
   *  If the mastergrid has a zero size, then the mastergrid is considered
   * to be infinitely large.
   */
  explicit World(real3 cellsize, Grid mastergrid = Grid(int3{0, 0, 0}));

  /** Destroy the world. */
  ~World();

  /** Return the current time of the World */
  real time() const;

  /** Return the cell size dimensions. */
  real3 cellsize() const;

  /** Return the cell volume. */
  real cellVolume() const;

  /** Return the master grid of the world.
   *  If the mastergrid has a zero size, then the mastergrid is considered to be
   *  infinitely large.
   */
  Grid mastergrid() const;

  /** Return the number of repetitions everything inside mastergrid in the x, y
   * and z directions to create periodic boundary conditions. The number of
   * repetitions determines the cutoff range for the demagnetization.
   * 
   * For example {2,0,1} means that, for the strayFieldKernel computation,
   * all magnets are essentially copied twice to the right, twice to the left,
   * but not in the y direction. That row is then copied once up and once down,
   * creating a 5x1x3 grid.
   */
  const int3 pbcRepetitions() const;

  /** Returns true if the grid is completely inside the mastergrid. */
  bool inMastergrid(Grid) const;

  /** Return a reference to the world's timesolver. */
  TimeSolver& timesolver() const;

 protected:
  real3 cellsize_;
  Grid mastergrid_;
  int3 pbcRepetitions_;
  std::unique_ptr<TimeSolver> timesolver_;
};
