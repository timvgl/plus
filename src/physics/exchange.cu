#include "cudalaunch.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool exchangeAssuredZero(const Ferromagnet* magnet) {
  return ((magnet->aex.assuredZero() && magnet->aex2.assuredZero()
          && magnet->afmex_cell.assuredZero() && magnet->afmex_nn.assuredZero())
          || (magnet->msat.assuredZero() && magnet->msat2.assuredZero()));
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  if (a == b)
    return a;
  return 2 * a * b / (a + b);
}

__global__ void k_exchangeField(CuField hField,
                                const CuField mField,
                                const CuParameter aex,
                                const CuParameter aex2,
                                const CuParameter afmex_cell,
                                const CuParameter afmex_nn,
                                const CuParameter msat,
                                const CuParameter msat2,
                                const CuParameter latcon,
                                const real3 w,  // w = 1/cellsize^2
                                Grid mastergrid,
                                const int comp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool afm = (comp == 6);

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx)) {
      if (comp == 3) {hField.setVectorInCell(idx, real3{0, 0, 0});}
      else if (comp == 6) {hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});}
    }
    return;
  }

  const Grid grid = mField.system.grid;

  if (!grid.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0 && msat2.valueAt(idx) == 0) {
    if (comp == 3) {hField.setVectorInCell(idx, real3{0, 0, 0});}
    else if (comp == 6) {hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});}
    return;
  }

  real6 m{0, 0, 0, 0, 0, 0};
  const int3 coo = grid.index2coord(idx);
  if (!afm) {
    const real3 mag = mField.FM_vectorAt(idx);
    m = {mag.x, mag.y, mag.z, 0, 0, 0};
  }
  else {
    m = mField.AFM_vectorAt(idx);
  }
  
  const real a = aex.valueAt(idx);
  const real a2 = aex2.valueAt(idx);
  const real ac = afmex_cell.valueAt(idx);
  const real ann = afmex_nn.valueAt(idx);

  // accumulate exchange field in h for cell at idx, divide by msat at the end
  // h is considered real6. 3 zeros are stripped at the end in case of FM.
  real6 h{0, 0, 0, 0, 0, 0};

  // AFM exchange at idx
  if (afm) {
    const real l = latcon.valueAt(idx);
    h += 4 * ac * real6{m.x2, m.y2, m.z2, m.x1, m.y1, m.z1} / (l * l);
  }
  
  // FM and AFM exchange in NN cells
  // X direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{sgn, 0, 0});
    if (!hField.cellInGeometry(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat.valueAt(idx_) != 0 || msat2.valueAt(idx_) != 0) {

      real6 m_{0, 0, 0, 0, 0, 0};
      if (!afm) {
        const real3 mag = mField.FM_vectorAt(idx_);
        m_ = {mag.x, mag.y, mag.z, 0, 0, 0};
      }
      else {
        m_ = mField.AFM_vectorAt(idx_);
      }

      const real a_ = aex.valueAt(idx_);
      const real a2_ = aex2.valueAt(idx_);
      const real ann_ = afmex_nn.valueAt(idx_);

      real6 lap = w.x * (m_ - m);

      // FM exchange
      h += 2 * real2{harmonicMean(a, a_), harmonicMean(a2, a2_)} * lap;
      // AFM exchange
      if (afm)
        h += harmonicMean(ann, ann_)
             * real6{lap.x2, lap.y2, lap.z2, lap.x1, lap.y1, lap.z1};
    }
  }

  // Y direction
#pragma unroll
  for (int sgn : {-1, 1}) {
    const int3 coo_ = mastergrid.wrap(coo + int3{0, sgn, 0});
    if (!hField.cellInGeometry(coo_))
      continue;
    const int idx_ = grid.coord2index(coo_);
    if (msat.valueAt(idx_) != 0 || msat2.valueAt(idx_) != 0) {

      real6 m_{0, 0, 0, 0, 0, 0};
      if (!afm) {
        const real3 mag = mField.FM_vectorAt(idx_);
        m_ = {mag.x, mag.y, mag.z, 0, 0, 0};
      }
      else {
        m_ = mField.AFM_vectorAt(idx_);
      }

      const real a_ = aex.valueAt(idx_);
      const real a2_ = aex2.valueAt(idx_);
      const real ann_ = afmex_nn.valueAt(idx_);

      real6 lap = w.y * (m_ - m);

      // FM exchange
      h += 2 * real2{harmonicMean(a, a_), harmonicMean(a2, a2_)} * lap;
      // AFM exchange
      if (afm)
        h += harmonicMean(ann, ann_)
             * real6{lap.x2, lap.y2, lap.z2, lap.x1, lap.y1, lap.z1};
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
      if (msat.valueAt(idx_) != 0 || msat2.valueAt(idx_) != 0) {

        real6 m_{0, 0, 0, 0, 0, 0};
        if (!afm) {
          const real3 mag = mField.FM_vectorAt(idx_);
          m_ = {mag.x, mag.y, mag.z, 0, 0, 0};
        }
        else {
          m_ = mField.AFM_vectorAt(idx_);
        }

        const real a_ = aex.valueAt(idx_);
        const real a2_ = aex2.valueAt(idx_);
        const real ann_ = afmex_nn.valueAt(idx_);

        real6 lap = w.z * (m_ - m);

        // FM exchange
        h += 2 * real2{harmonicMean(a, a_), harmonicMean(a2, a2_)} * lap;
        // AFM exchange
        if (afm)
          h += harmonicMean(ann, ann_)
              * real6{lap.x2, lap.y2, lap.z2, lap.x1, lap.y1, lap.z1};
      }
    }
  }

  //  int3 neighborRelativeCoordinat__restrict__ es[6] = {int3{-1, 0, 0},
  //  int3{0, -1, 0},
  //                                         int3{0, 0, -1}, int3{1, 0, 0},
  //                                         int3{0, 1, 0},  int3{0, 0, 1}};
  //
  //  #pragma unroll
  //  for (int3 relcoo : neighborRelativeCoordinates) {
  //    const int3 coo_ = coo + relcoo;
  //    const int idx_ = hField.grid.coord2index(coo_);
  //
  //    if (hField.cellInGrid(coo_) && msat.valueAt(idx_) != 0) {
  //      real dr =
  //          cellsize.x * relcoo.x + cellsize.y * relcoo.y + cellsize.z *
  //          relcoo.z;
  //      real3 m_ = mField.vectorAt(idx_);
  //      real a_ = aex.valueAt(idx_);
  //
  //      h += 2 * harmonicMean(a, a_) * (m_ - m) / (dr * dr);
  //    }
  //  }

  if (!afm) {
    real3 heff = real3{h.x1, h.y1, h.z1} / msat.valueAt(idx);
    hField.setVectorInCell(idx, heff);
  }
  else {
    h /= real2{msat.valueAt(idx), msat2.valueAt(idx)};
    hField.setVectorInCell(idx, h);
  }
}

Field evalExchangeField(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->ncomp();
  Field hField(magnet->system(), comp);
  
  if (exchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  real3 c = magnet->cellsize();
  real3 w = {1 / (c.x * c.x), 1 / (c.y * c.y), 1 / (c.z * c.z)};
  
  auto magnetization = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto msat2 = magnet->msat2.cu();
  auto aex = magnet->aex.cu();
  auto aex2 = magnet->aex2.cu();
  auto afmex_cell = magnet->afmex_cell.cu();
  auto afmex_nn = magnet->afmex_nn.cu();
  auto latcon = magnet->latcon.cu();

  cudaLaunch(hField.grid().ncells(), k_exchangeField, hField.cu(),
             magnetization, aex, aex2, afmex_cell, afmex_nn, msat, msat2,
             latcon, w, magnet->world()->mastergrid(), comp);
  return hField;
}

Field evalExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (exchangeAssuredZero(magnet))
    return Field(magnet->system(), magnet->magnetization()->ncomp() / 3, 0.0);
  return evalEnergyDensity(magnet, evalExchangeField(magnet), 0.5);
}

real evalExchangeEnergy(const Ferromagnet* magnet, const bool sub2) {
  if (exchangeAssuredZero(magnet))
    return 0;
    
  real edens;
  if (!sub2) 
    edens = exchangeEnergyDensityQuantity(magnet).average()[0];
  else 
    edens = exchangeEnergyDensityQuantity(magnet).average()[1];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity exchangeFieldQuantity(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->field().cu().ncomp;
  return FM_FieldQuantity(magnet, evalExchangeField, comp, "exchange_field", "T");
}

FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalExchangeEnergyDensity, comp / 3,
                          "exchange_energy_density", "J/m3");
}

FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  std::string name = sub2 ? "exchange_energy2" : "exchange_energy";
  return FM_ScalarQuantity(magnet, evalExchangeEnergy, sub2, name, "J");
}

__global__ void k_maxangle(CuField maxAngleField,
                           const CuField mField,
                           const CuParameter aex,
                           const CuParameter aex2,
                           const CuParameter msat,
                           const CuParameter msat2,
                           const int comp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!maxAngleField.cellInGeometry(idx)) {
    if (maxAngleField.cellInGrid(idx)) {
      if(comp == 3) 
        maxAngleField.setValueInCell(idx, 0, 0);
      else if (comp == 6)
        maxAngleField.setValueInCell(idx, 0, 0);
    }
    return;
  }

  const Grid grid = maxAngleField.system.grid;

  if (msat.valueAt(idx) == 0 && msat2.valueAt(idx) == 0) {
    maxAngleField.setValueInCell(idx, 0, 0);
    return;
  }

  const int3 coo = grid.index2coord(idx);
  const real a = aex.valueAt(idx);
  const real a2 = aex2.valueAt(idx);


  real maxAngle{0};  // maximum angle in this cell

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

#pragma unroll
  for (int3 relcoo : neighborRelativeCoordinates) {
    const int3 coo_ = coo + relcoo;
    const int idx_ = grid.coord2index(coo_);
    if (mField.cellInGeometry(coo_) && (msat.valueAt(idx_) != 0 || msat2.valueAt(idx_) != 0)) {
      real a_ = aex.valueAt(idx_);
      real a2_ = aex2.valueAt(idx_);
      if (comp == 3) {
        real3 m = mField.FM_vectorAt(idx);
        real3 m_ = mField.FM_vectorAt(idx_);
        real angle = m == m_ ? 0 : acos(dot(m, m_));
        if (harmonicMean(a, a_) != 0 && angle > maxAngle) {
          maxAngle = angle;
        }
      }
      else if (comp == 6){
        real6 m = mField.AFM_vectorAt(idx);
        real6 m_ = mField.AFM_vectorAt(idx_);
        real anglex = m == m_ ? 0 : acos(dot(m, m_).x);
        real angley = m == m_ ? 0 : acos(dot(m, m_).y);
        if (real2{harmonicMean(a, a_), harmonicMean(a2, a2_)} != real2{0, 0}
            && (anglex > maxAngle || angley > maxAngle)) {
          if (anglex > angley) {maxAngle = anglex;}
          else {maxAngle = angley;}
        }
      }
    }
  }

  maxAngleField.setValueInCell(idx, 0, maxAngle);
}

real evalMaxAngle(const Ferromagnet* magnet, const bool sub2) {
  Field maxAngleField(magnet->system(), 1);
  cudaLaunch(maxAngleField.grid().ncells(), k_maxangle, maxAngleField.cu(),
             magnet->magnetization()->field().cu(), magnet->aex.cu(), magnet->aex2.cu(),
             magnet->msat.cu(), magnet->msat2.cu(), magnet->magnetization()->ncomp());
  return maxAbsValue(maxAngleField);
}

FM_ScalarQuantity maxAngle(const Ferromagnet* magnet, const bool sub2) {
  return FM_ScalarQuantity(magnet, evalMaxAngle, sub2, "max_angle", "");
}
