#include "antiferromagnet.hpp"
#include "butchertableau.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include "relaxer.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

#include <algorithm>

Relaxer::Relaxer(const Magnet* magnet, std::vector<real> RelaxTorqueThreshold)
    : magnet_(magnet),
      threshold_(RelaxTorqueThreshold) {}

std::vector<FM_FieldQuantity> Relaxer::getTorque() {
  if (const Ferromagnet* mag = magnet_->asFM())
    return {relaxTorqueQuantity(mag)};
  else if (const Antiferromagnet* mag = magnet_->asAFM())
    return {relaxTorqueQuantity(mag->sub1()),
            relaxTorqueQuantity(mag->sub2())};
  else
    throw std::invalid_argument("Cannot relax quantity which is"
                                "no Ferromagnet or Antiferromagnet.");
}

void Relaxer::exec() {

  TimeSolver &timesolver = magnet_->world()->timesolver();
  
  // Store current solver settings
  real time = timesolver.time();
  real timestep = timesolver.timestep();
  bool adaptive = timesolver.hasAdaptiveTimeStep();
  real maxerr = timesolver.maxerror();
  bool prec = timesolver.hasPrecession();
  std::string method = getRungeKuttaNameFromMethod(timesolver.getRungeKuttaMethod());

  // Set solver settings for relax
  timesolver.disablePrecession();
  timesolver.enableAdaptiveTimeStep();
  timesolver.setRungeKuttaMethod("BogackiShampine");

  // Run while monitoring energy
  const int N = 3; // evaluates energy every N steps (expenisve)  

  real E0 = evalTotalEnergy(magnet_);
  timesolver.steps(N);
  real E1 = evalTotalEnergy(magnet_);
  while (E1 < E0) {
    timesolver.steps(N);
    E0 = E1;
    E1 = evalTotalEnergy(magnet_);
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

    real err = timesolver.maxerror();
    while (err > 1e-9) {
      err /= std::sqrt(2);
      timesolver.setMaxError(err);

      timesolver.steps(N);

      bool torqueConverged = false;
      while(!torqueConverged) {
        for (size_t i = 0; i < torque.size(); i++) {
          t0[i] = t1[i];
          t1[i] = dotSum(torque[i].eval(), torque[i].eval());
        }

        if (converged(t1, t0)) {
          break;
        }
        timesolver.steps(N);
      }     
    }
  }

  else if (std::find(threshold_.begin(), threshold_.end(), 0) != threshold_.end())
    throw std::invalid_argument("The relax threshold should not be zero.");

  // If threshold is set by user: relax until torque is smaller than or equal to threshold.
  else { 
    real err = timesolver.maxerror();
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
        timesolver.setMaxError(err);
        timesolver.steps(N);
      }

      else { timesolver.steps(N); }
    }
  }

  // Restore solver settings after relaxing
  timesolver.setRungeKuttaMethod(method);
  timesolver.setMaxError(maxerr);
  if (!adaptive) { timesolver.disableAdaptiveTimeStep(); }
  if (prec) { timesolver.enablePrecession(); }
  timesolver.setTime(time);
  timesolver.setTimeStep(timestep); 
}

bool Relaxer::converged(std::vector<real> t1, std::vector<real> t2) {
 for (size_t i = 0; i < t1.size(); ++i) {
        if (t1[i] < t2[i]) { return false; }
    }
    return true;
}