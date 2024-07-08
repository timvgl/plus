#include "antiferromagnet.hpp"
#include "butchertableau.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "fieldops.hpp"
#include "mumaxworld.hpp"
#include "reduce.hpp"
#include "relaxer.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

#include <algorithm>

Relaxer::Relaxer(const Magnet* magnet, std::vector<real> RelaxTorqueThreshold)
    : magnets_({magnet}),
      timesolver_(magnet->world()->timesolver()),
      threshold_(RelaxTorqueThreshold) {}

Relaxer::Relaxer(const MumaxWorld* world)
    : world_(world),
      timesolver_(world->timesolver()) {
        for (const auto& pair : world->magnets())
          magnets_.push_back(pair.second);
}

std::vector<FM_FieldQuantity> Relaxer::getTorque() {
  std::vector<FM_FieldQuantity> torque;
  for (auto magnet : magnets_) {
    if (const Ferromagnet* mag = magnet->asFM())
     torque.push_back(relaxTorqueQuantity(mag));
    else if (const Antiferromagnet* mag = magnet->asAFM())
      torque.push_back(relaxTorqueQuantity(mag->sub1()));
    else
      throw std::invalid_argument("Cannot relax quantity which is"
                                  "no Ferromagnet or Antiferromagnet.");
  }
  return torque;
}

real Relaxer::calcEnergy() {
  real E = 0;
  for (auto magnet : magnets_)
    E += evalTotalEnergy(magnet);
  return E;
}

void Relaxer::exec() {
  
  // Store current solver settings
  real time = timesolver_.time();
  real timestep = timesolver_.timestep();
  bool adaptive = timesolver_.hasAdaptiveTimeStep();
  real maxerr = timesolver_.maxerror();
  bool prec = timesolver_.hasPrecession();
  std::string method = getRungeKuttaNameFromMethod(timesolver_.getRungeKuttaMethod());

  // Set solver settings for relax
  timesolver_.disablePrecession();
  timesolver_.enableAdaptiveTimeStep();
  timesolver_.setRungeKuttaMethod("BogackiShampine");

  // Run while monitoring energy
  const int N = 3; // evaluates energy every N steps (expenisve)  

  real E0 = calcEnergy();
  timesolver_.steps(N);
  real E1 = calcEnergy();
  while (E1 < E0) {
    timesolver_.steps(N);
    E0 = E1;
    E1 = calcEnergy();
  }
  
  // Run while monitoring torque
  // If threshold = -1 (default) or < 0: relax until torque is steady or increasing.
  if (std::all_of(threshold_.begin(), threshold_.end(), [](real t) { return t < 0; })) {

    std::vector<FM_FieldQuantity> torque = getTorque();
    std::vector<real> t0(torque.size(), 0);
    std::vector<real> t1(torque.size());
    for (size_t i = 0; i < torque.size(); i++) {
      t1[i] = dotSum(torque[i].eval(), torque[i].eval());
    }

    real err = timesolver_.maxerror();
    while (err > 1e-9) {
      err /= std::sqrt(2);
      timesolver_.setMaxError(err);

      timesolver_.steps(N);

      bool torqueConverged = false;
      while(!torqueConverged) {
        for (size_t i = 0; i < torque.size(); i++) {
          t0[i] = t1[i];
          t1[i] = dotSum(torque[i].eval(), torque[i].eval());
        }

        if (converged(t0, t1)) { break; }
        timesolver_.steps(N);
      }     
    }
  }

  else if (std::find(threshold_.begin(), threshold_.end(), 0) != threshold_.end())
    throw std::invalid_argument("The relax threshold should not be zero.");

  // If threshold is set by user: relax until torque is smaller than or equal to threshold.
  else { 
    real err = timesolver_.maxerror();
    std::vector<FM_FieldQuantity> torque = getTorque();

    while (err > 1e-9) {
      bool torqueConverged = true;
      for (size_t i = 0; i < torque.size(); i++) {
        if (maxVecNorm(torque[i].eval()) > threshold_[i]) {
          torqueConverged = false;
          break;
        }
      }

      if (torqueConverged) {    
        err /= std::sqrt(2);
        timesolver_.setMaxError(err);
      }
      timesolver_.steps(N);
    }
  }

  // Restore solver settings after relaxing
  timesolver_.setRungeKuttaMethod(method);
  timesolver_.setMaxError(maxerr);
  if (!adaptive) { timesolver_.disableAdaptiveTimeStep(); }
  if (prec) { timesolver_.enablePrecession(); }
  timesolver_.setTime(time);
  timesolver_.setTimeStep(timestep); 
}

bool Relaxer::converged(std::vector<real> t0, std::vector<real> t1) {
 for (size_t i = 0; i < t1.size(); ++i) {
        if (t1[i] < t0[i]) { return false; }
    }
    return true;
}