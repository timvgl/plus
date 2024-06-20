#include "ferromagnet.hpp"

#include <curand.h>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "poissonsystem.hpp"
#include "system.hpp"

Ferromagnet::Ferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry)
    : Magnet(world, grid, name, geometry),
      magnetization_(name + ":magnetization", "", system(), 3),
      msat(system(), 1.0),
      aex(system(), 0.0),
      ku1(system(), 0.0),
      ku2(system(), 0.0),
      kc1(system(), 0.0),
      kc2(system(), 0.0),
      kc3(system(), 0.0),
      alpha(system(), 0.0),
      temperature(system(), 0.0),
      idmi(system(), 0.0),
      xi(system(), 0.0),
      Lambda(system(), 0.0),
      FreeLayerThickness(system(), grid.size().z * cellsize().z),
      eps_prime(system(), 0.0),
      FixedLayer(system(), {0, 0, 0}),
      pol(system(), 0.0),
      anisU(system(), {0, 0, 0}),
      anisC1(system(), {0, 0, 0}),
      anisC2(system(), {0, 0, 0}),
      jcur(system(), {0, 0, 0}),
      biasMagneticField(system(), {0, 0, 0}),
      dmiTensor(system()),
      enableDemag(true),
      enableOpenBC(false),
      appliedPotential(system(), std::nanf("0")),
      conductivity(system(), 0.0),
      amrRatio(system(), 0.0),
      poissonSystem(this) {
    {
    // TODO: this can be done much more efficient somewhere else
    int nvalues = 3 * this->grid().ncells();
    std::vector<real> randomValues(nvalues);
    std::normal_distribution<real> dist(0.0, 1.0);
    std::default_random_engine randomEngine;
    for (auto& v : randomValues) {
      v = dist(randomEngine);
    }
    Field randomField(this->system(), 3);
    randomField.setData(&randomValues[0]);
    magnetization_.set(randomField);
  }
  // TODO: move the generator to somewhere else
  curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randomGenerator, 1234);
  }

Ferromagnet::~Ferromagnet() {
  for (auto& entry : strayFields_) {
    delete entry.second;
  }
  curandDestroyGenerator(randomGenerator);
}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}