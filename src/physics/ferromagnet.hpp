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
#include "magnet.hpp"
#include "parameter.hpp"
#include "poissonsystem.hpp"
#include "variable.hpp"
#include "world.hpp"

class Ferromagnet : public Magnet {
 public:
  Ferromagnet(MumaxWorld* world,
        Grid grid,
         std::string name,
         GpuBuffer<bool> geometry);
  ~Ferromagnet() override;

  const Variable* magnetization() const;

  void minimize(real tol = 1e-6, int nSamples = 10);

 private:
  NormalizedVariable magnetization_;

 public:
  mutable PoissonSystem poissonSystem;
  bool enableDemag;
  bool enableOpenBC;
  FM_VectorParameter anisU;
  FM_VectorParameter anisC1;
  FM_VectorParameter anisC2;
  FM_VectorParameter jcur;
  FM_VectorParameter FixedLayer;
  /** Uniform bias magnetic field which will affect a ferromagnet.
   * Measured in Teslas.
   */
  FM_VectorParameter biasMagneticField;
  Parameter msat;
  Parameter aex;
  Parameter ku1;
  Parameter ku2;
  Parameter kc1;
  Parameter kc2;
  Parameter kc3;
  Parameter alpha;
  Parameter temperature;
  Parameter idmi;
  Parameter Lambda;
  Parameter FreeLayerThickness;
  Parameter eps_prime;
  Parameter xi;
  Parameter pol;
  Parameter appliedPotential;
  Parameter conductivity;
  Parameter amrRatio;

  curandGenerator_t randomGenerator;

  DmiTensor dmiTensor;
};
