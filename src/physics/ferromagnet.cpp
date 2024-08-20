#include "ferromagnet.hpp"

#include <curand.h>
#include <chrono>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "poissonsystem.hpp"
#include "relaxer.hpp"

Ferromagnet::Ferromagnet(std::shared_ptr<System> system_ptr, 
                         std::string name,
                         Antiferromagnet* hostMagnet)
    : Magnet(system_ptr, name),
      hostMagnet_(hostMagnet),
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
      FreeLayerThickness(system(), grid().size().z * cellsize().z),
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
      enableZhangLiTorque(true),
      enableSlonczewskiTorque(true),
      enableElastodynamics_(false),
      appliedPotential(system(), std::nanf("0")),
      conductivity(system(), 0.0),
      amrRatio(system(), 0.0),
      RelaxTorqueThreshold(-1.0),
      poissonSystem(this), 
      // magnetoelasticity
      c11(system(), 0.0),
      c12(system(), 0.0),
      c44(system(), 0.0),
      eta(system(), 0.0),
      rho(system(), 1.0),  // TODO: different default?
      B1(system(), 0.0),
      B2(system(), 0.0) {
    {
    // TODO: this can be done much more efficient somewhere else
    int nvalues = 3 * this->grid().ncells();
    std::vector<real> randomValues(nvalues);
    std::normal_distribution<real> dist(0.0, 1.0);
    std::default_random_engine randomEngine;
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    randomEngine.seed (seed);
    for (auto& v : randomValues) {
      v = dist(randomEngine);
    }
    Field randomField(system(), 3);
    randomField.setData(&randomValues[0]);
    magnetization_.set(randomField);
  }
  // TODO: move the generator to somewhere else
  curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randomGenerator,
        static_cast<int>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  }
Ferromagnet::Ferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry)
    : Ferromagnet(std::make_shared<System>(world, grid, geometry), name) {}

Ferromagnet::~Ferromagnet() {
  curandDestroyGenerator(randomGenerator);
}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

const Variable* Ferromagnet::elasticDisplacement() const {
  return elasticDisplacement_.get();
}

const Variable* Ferromagnet::elasticVelocity() const {
  return elasticVelocity_.get();
}

bool Ferromagnet::isSublattice() const {
  return !(hostMagnet_ == nullptr);
}

const Antiferromagnet* Ferromagnet::hostMagnet() const {
  return hostMagnet_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Ferromagnet::relax(real tol) {
  real threshold = this->RelaxTorqueThreshold;
  Relaxer relaxer(this, {threshold}, tol);
  relaxer.exec();
}

void Ferromagnet::setEnableElastodynamics(bool value) {
  enableElastodynamics_ = value;

  if (value) {
    // properly initialize Variables now
    elasticDisplacement_ = std::make_unique<Variable>(
                             name() + ":elasticDisplacement", "m", system(), 3);
    elasticVelocity_ = std::make_unique<Variable>(
                               name() + ":elasticVelocity", "m/s", system(), 3);
  } else {
    // free memory of unnecessary Variables
    elasticDisplacement_.reset();
    elasticVelocity_.reset();
  }

  this->mumaxWorld()->resetTimeSolverEquations();
}
