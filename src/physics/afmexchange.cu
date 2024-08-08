#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp" // used for Neumann BC
#include "energy.hpp"
#include "afmexchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool afmExchangeAssuredZero(const Ferromagnet* magnet) {
  if (!magnet->isSublattice()) { return true; }

  return ((magnet->hostMagnet()->afmex_cell.assuredZero()
        && magnet->hostMagnet()->afmex_nn.assuredZero())
        || (magnet->msat.assuredZero()
        && magnet->hostMagnet()->getOtherSublattice(magnet)->msat.assuredZero()));
}

__global__ void k_afmExchangeField(CuField hField,
                                const CuField m1Field,
                                const CuField m2Field,
                                const CuParameter aex,
                                const CuParameter afmex_cell,
                                const CuParameter afmex_nn,
                                const CuParameter msat2,
                                const CuParameter latcon,
                                const real3 w,  // w = 1/cellsize^2
                                const Grid mastergrid,
                                const CuDmiTensor dmiTensor,
                                bool openBC) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = m2Field.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat2.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real3 m2 = m2Field.vectorAt(idx);
  const real a = aex.valueAt(idx);
  const real ac = afmex_cell.valueAt(idx);
  const real ann = afmex_nn.valueAt(idx);
  
  // If there is no FM-exchange at the boundary, open BC are assumed
  openBC = (a == 0) ? true : openBC;

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};

  // AFM exchange at idx
  const real l = latcon.valueAt(idx);
  h += 4 * ac * m2 / (l * l);
  
  // AFM exchange in NN cells
#pragma unroll
  for (int3 rel_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 coo_ = mastergrid.wrap(coo + rel_coo);

    if(!hField.cellInGeometry(coo_) && openBC)
      continue;
    
    const int idx_ = grid.coord2index(coo_);
    real delta = dot(rel_coo, system.cellsize);

    if(msat2.valueAt(idx_) != 0 || !openBC) {
      real3 m2_;
      real ann_;
      int3 normal = rel_coo * rel_coo;

      if(hField.cellInGeometry(coo_)) {
        m2_ = m2Field.vectorAt(idx_);
        ann_ = afmex_nn.valueAt(idx_);
      }
      else { // Neumann BC
      real3 Gamma1 = getGamma(dmiTensor, idx, normal, m1Field.vectorAt(idx));
      real fac = ann / (2 * a);
      if(fac == -1) {
        m2_ = m2 + Gamma1 / (4*a) * delta;
      }
      else {
        real3 Gamma2 = getGamma(dmiTensor, idx, normal, m2);
        m2_ = m2 + delta / (a * 2 * (1 - fac*fac)) * (Gamma2 - fac * Gamma1);
      }
      ann_ = ann;
      }
      h += harmonicMean(ann, ann_) * dot(normal, w) * (m2_ - m2);
    }
  }
  hField.setVectorInCell(idx, h / msat2.valueAt(idx));
}

Field evalAFMExchangeField(const Ferromagnet* magnet) {

  Field hField(magnet->system(), 3);
  
  if (afmExchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  real3 c = magnet->cellsize();
  real3 w = {1 / (c.x * c.x), 1 / (c.y * c.y), 1 / (c.z * c.z)};
  
  auto otherSub = magnet->hostMagnet()->getOtherSublattice(magnet);
  auto mag = magnet->magnetization()->field().cu();
  auto otherMag = otherSub->magnetization()->field().cu();
  auto msat2 = otherSub->msat.cu();
  auto aex = magnet->aex.cu();
  auto afmex_cell = magnet->hostMagnet()->afmex_cell.cu();
  auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
  auto latcon = magnet->hostMagnet()->latcon.cu();
  auto BC = magnet->enableOpenBC;
  auto dmiTensor = magnet->dmiTensor.cu();

  cudaLaunch(hField.grid().ncells(), k_afmExchangeField, hField.cu(),
            mag, otherMag, aex, afmex_cell, afmex_nn, msat2, latcon,
            w, magnet->world()->mastergrid(), dmiTensor, BC);
  return hField;
}

Field evalAFMExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (afmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalAFMExchangeField(magnet), 0.5);
}

real evalAFMExchangeEnergy(const Ferromagnet* magnet) {
  if (afmExchangeAssuredZero(magnet))
    return 0;
    
  real edens = AFMexchangeEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity AFMexchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAFMExchangeField, 3, "exchange_field", "T");
}

FM_FieldQuantity AFMexchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAFMExchangeEnergyDensity, 1,
                          "afm_exchange_energy_density", "J/m3");
}

FM_ScalarQuantity AFMexchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalAFMExchangeEnergy, "afm_exchange_energy", "J");
}

__global__ void k_angle(CuField angleField,
                        const CuField mField1,
                        const CuField mField2,
                        const CuParameter afmex,
                        const CuParameter msat1,
                        const CuParameter msat2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!angleField.cellInGeometry(idx)) {
    if (angleField.cellInGrid(idx)) 
      angleField.setValueInCell(idx, 0, 0);
    return;
  }

  if (msat1.valueAt(idx) == 0 || msat2.valueAt(idx) == 0 || afmex.valueAt(idx) == 0) {
    angleField.setValueInCell(idx, 0, 0);
    return;
  }

  angleField.setValueInCell(idx, 0, acos(copysign(1.0, afmex.valueAt(idx))
                                            * dot(mField1.vectorAt(idx),
                                                  mField2.vectorAt(idx))));
}

Field evalAngleField(const Antiferromagnet* magnet) {
  Field angleField(magnet->system(), 1);

  cudaLaunch(angleField.grid().ncells(), k_angle, angleField.cu(),
            magnet->sub1()->magnetization()->field().cu(),
            magnet->sub2()->magnetization()->field().cu(),
            magnet->afmex_cell.cu(),
            magnet->sub1()->msat.cu(), magnet->sub2()->msat.cu());
  return angleField;
}

real evalMaxAngle(const Antiferromagnet* magnet) {
  return maxAbsValue(evalAngleField(magnet));
}

AFM_FieldQuantity angleFieldQuantity(const Antiferromagnet* magnet) {
  return AFM_FieldQuantity(magnet, evalAngleField, 1, "angle_field", "");
}

AFM_ScalarQuantity maxAngle(const Antiferromagnet* magnet) {
  return AFM_ScalarQuantity(magnet, evalMaxAngle, "max_angle", "");
}