#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "dmi.hpp"
#include "dmitensor.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool dmiAssuredZero(const Ferromagnet* magnet) {
  return (magnet->dmiTensor.assuredZero() || magnet->msat.assuredZero());
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  return 2 * a * b / (a + b);
}

__device__ static inline real harmonicMean(const CuParameter& param,
                                           int idx1,
                                           int idx2) {
  return harmonicMean(param.valueAt(idx1), param.valueAt(idx2));
}

__global__ void k_dmiField(CuField hField,
                           const CuField mField,
                           const CuDmiTensor dmiTensor,
                           const CuParameter msat,
                           Grid mastergrid,
                           const CuParameter aex,
                           const bool enableOpenBC,
                           const bool Dint,
                           const bool Dbulk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

	real3 m = mField.vectorAt(idx);
  const int3 coo = system.grid.index2coord(idx);

  // Accumulate DMI field of cell at idx in h. Divide by msat at the end.
  real3 h{0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx = system.grid.coord2index(neighbor_coo);

    // If we assume open boundary conditions and if there is no neighbor, 
    // then simply continue without adding to the effective field.    
    if ((!system.inGeometry(neighbor_coo) && enableOpenBC) || 
        ((msat.valueAt(neighbor_idx) == 0) && enableOpenBC))
      continue;
    
    
    // Get the dmi strengths between the center cell and the neighbor, which are
    // the harmonic means of the dmi strengths of both cells.
    real Dxz, Dxy, Dyz, Dzx, Dyx, Dzy;

    if(system.inGeometry(neighbor_coo)) {
      if (relative_coo.x) {  // derivative along x
        Dxz = harmonicMean(dmiTensor.xxz, idx, neighbor_idx);
        Dxy = harmonicMean(dmiTensor.xxy, idx, neighbor_idx);
        Dyz = harmonicMean(dmiTensor.xyz, idx, neighbor_idx);
      } else if (relative_coo.y) {  // derivative along y
        Dxz = harmonicMean(dmiTensor.yxz, idx, neighbor_idx);
        Dxy = harmonicMean(dmiTensor.yxy, idx, neighbor_idx);
        Dyz = harmonicMean(dmiTensor.yyz, idx, neighbor_idx);
      } else if (relative_coo.z) {  // derivative along z
        Dxz = harmonicMean(dmiTensor.zxz, idx, neighbor_idx);
        Dxy = harmonicMean(dmiTensor.zxy, idx, neighbor_idx);
        Dyz = harmonicMean(dmiTensor.zyz, idx, neighbor_idx);
      }
    }
    else {// Used for DMI-BC
      if (relative_coo.x) {  // derivative along x
        Dxz = dmiTensor.xxz.valueAt(idx);
        Dxy = dmiTensor.xxy.valueAt(idx);
        Dyz = dmiTensor.xyz.valueAt(idx);
      } else if (relative_coo.y) {  // derivative along y
        Dxz = dmiTensor.yxz.valueAt(idx);
        Dxy = dmiTensor.yxy.valueAt(idx);
        Dyz = dmiTensor.yyz.valueAt(idx);
      } else if (relative_coo.z) {  // derivative along z
        Dxz = dmiTensor.zxz.valueAt(idx);
        Dxy = dmiTensor.zxy.valueAt(idx);
        Dyz = dmiTensor.zyz.valueAt(idx);
      }
    }

    Dzx = -Dxz;  // dmi tensor is assymetric
    Dyx = -Dxy;
    Dzy = -Dyz;

    // Distance between neighbors (the finite difference)
    real delta = relative_coo.x * system.cellsize.x +
                 relative_coo.y * system.cellsize.y +
                 relative_coo.z * system.cellsize.z;

    real3 m_;
    
    if (!system.inGeometry(neighbor_coo) && !enableOpenBC) {
    // DMI-BC
      real a = aex.valueAt(idx);
      real D = Dxy + Dxz + Dyz - 2 * Dbulk * Dxz;
      real D_2A = D / (2 * a);
      real Ax = D_2A * system.cellsize.x * relative_coo.x;
      real Ay = D_2A * system.cellsize.y * relative_coo.y;
      real Az = D_2A * system.cellsize.z * relative_coo.z;

      if (relative_coo.x) {
        m_.x = m.x - Dint  * (Ax * m.z);
        m_.y = m.y - Dbulk * (Ax * m.z);
        m_.z = m.z + Dbulk * (Ax * m.y) + Dint * (Ax * m.x);
      }
      else if (relative_coo.y) {
        m_.x = m.x + Dbulk * (Ay * m.z);
        m_.y = m.y - Dint  * (Ay * m.z);
        m_.z = m.z - Dbulk * (Ay * m.x) + Dint * (Ay * m.y);
      }
      else if (relative_coo.z) {
        m_.x = m.x - Dbulk * (Az * m.y);
        m_.y = m.y + Dbulk * (Az * m.x);
        m_.z = m.z;
      }
    }
    else {
      m_ = mField.vectorAt(neighbor_idx);
    }

    
    // Compute the effective field contribution of the DMI with the neighbor
    h.x += (Dxy * m_.y + Dxz * m_.z) / delta;
    h.y += (Dyx * m_.x + Dyz * m_.z) / delta;
    h.z += (Dzx * m_.x + Dzy * m_.y) / delta;

  }  // end loop over neighbors

  /* TO DO: implement DMI-contribution to afm-exchange at a single site
  if (comp == 6) {
    // Compute effective field contribution of the DMI between sublattices at a single site
    real Dyz = dmiTensor.xyz.valueAt(idx);
    real Dxz = dmiTensor.yxz.valueAt(idx);
    real Dxy = dmiTensor.zxy.valueAt(idx);
    real Dzx = -Dxz;
    real Dyx = -Dxy;
    real Dzy = -Dyz;
    real l = latcon.valueAt(idx);
    h.x1 += (Dxy * m.y2 + Dxz * m.z2) / l;
    h.y1 += (Dyx * m.x2 + Dyz * m.z2) / l;
    h.z1 += (Dzx * m.x2 + Dzy * m.y2) / l;
    h.x2 += (Dxy * m.y1 + Dxz * m.z1) / l;
    h.y2 += (Dyx * m.x1 + Dyz * m.z1) / l;
    h.z2 += (Dzx * m.x1 + Dzy * m.y1) / l;
  }*/

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);

}

Field evalDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (dmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  
  cudaLaunch(hField.grid().ncells(), k_dmiField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->dmiTensor.cu(),
             magnet->msat.cu(), magnet->world()->mastergrid(),
             magnet->aex.cu(), magnet->enableOpenBC,
             magnet->dmiTensor.isInterfacial(),
             magnet->dmiTensor.isBulk());
  return hField;
}

Field evalDmiEnergyDensity(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);

  return evalEnergyDensity(magnet, evalDmiField(magnet), 0.5);
}

real evalDmiEnergy(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return 0;

  real edens = dmiEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity dmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDmiField, 3, "dmi_field", "T");
}

FM_FieldQuantity dmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDmiEnergyDensity, 1, "dmi_emergy_density",
                          "J/m3");
}

FM_ScalarQuantity dmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalDmiEnergy, "dmi_energy", "J");
}
