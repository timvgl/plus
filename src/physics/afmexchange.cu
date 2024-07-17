#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
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

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  if (a == b)
    return a;
  return 2 * a * b / (a + b);
}

__global__ void k_afmExchangeField(CuField hField,
                                const CuField mField,
                                const CuParameter afmex_cell,
                                const CuParameter afmex_nn,
                                const CuParameter msat2,
                                const CuParameter latcon,
                                const real3 w,  // w = 1/cellsize^2
                                Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const Grid grid = mField.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat2.valueAt(idx) == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);


  const real3 m2 = mField.vectorAt(idx);

  const real ac = afmex_cell.valueAt(idx);
  const real ann = afmex_nn.valueAt(idx);

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  real3 h{0, 0, 0};

  // AFM exchange at idx
  const real l = latcon.valueAt(idx);
  h += 4 * ac * m2 / (l * l);
  
  // AFM exchange in NN cells
  // X direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{sgn, 0, 0});
    if (!hField.cellInGeometry(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat2.valueAt(idx_) != 0) {
      const real3 m2_ = mField.vectorAt(idx_);
      const real ann_ = afmex_nn.valueAt(idx_);
      h += harmonicMean(ann, ann_) * w.x * (m2_ - m2);
    }
  }

  // Y direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, sgn, 0});
    if (!hField.cellInGeometry(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat2.valueAt(idx_) != 0) {
      const real3 m2_ = mField.vectorAt(idx_);
      const real ann_ = afmex_nn.valueAt(idx_);
      h += harmonicMean(ann, ann_) * w.y * (m2_ - m2);
    }
  }

  // Z direction
  if (grid.size().z > 1) {
#pragma unroll
    for (int sgn : {-1, 1}) {
      const int3 coo_ = mastergrid.wrap(coo + int3{0, 0, sgn});
      if (!hField.cellInGeometry(coo_))
        continue;
      const int idx_ = grid.coord2index(coo_);
      if (msat2.valueAt(idx_) != 0) {
        const real3 m2_ = mField.vectorAt(idx_);
        const real ann_ = afmex_nn.valueAt(idx_);
        h += harmonicMean(ann, ann_) * w.z * (m2_ - m2);
      }
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
  auto otherMag = otherSub->magnetization()->field().cu();
  auto msat2 = otherSub->msat.cu();
  auto afmex_cell = magnet->hostMagnet()->afmex_cell.cu();
  auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
  auto latcon = magnet->hostMagnet()->latcon.cu();

  cudaLaunch(hField.grid().ncells(), k_afmExchangeField, hField.cu(), otherMag,
             afmex_cell, afmex_nn, msat2, latcon, w, magnet->world()->mastergrid());
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

  if (msat1.valueAt(idx) == 0 || msat2.valueAt(idx) == 0
                              || afmex.valueAt(idx) == 0) {
    angleField.setValueInCell(idx, 0, 0);
    return;
  }

  angleField.setValueInCell(idx, 0, acos(dot(mField1.vectorAt(idx),
                                                mField2.vectorAt(idx))));

  /*real maxAngle{0};  // maximum angle in this cell

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

#pragma unroll
  for (int3 relcoo : neighborRelativeCoordinates) {
    const int3 coo_ = coo + relcoo;
    const int idx_ = grid.coord2index(coo_);
    if (mField.cellInGeometry(coo_) && msat.valueAt(idx_) != 0) {
      real a_ = aex.valueAt(idx_);
      real3 m = mField.vectorAt(idx);
      real3 m_ = mField.vectorAt(idx_);
      real angle = m == m_ ? 0 : acos(dot(m, m_));
      if (harmonicMean(a, a_) != 0 && angle > maxAngle)
        maxAngle = angle;
    }
  }
  maxAngleField.setValueInCell(idx, 0, maxAngle);*/
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