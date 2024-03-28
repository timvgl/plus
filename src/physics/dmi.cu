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
  return (magnet->dmiTensor.assuredZero()
         || (magnet->msat.assuredZero() && magnet->msat2.assuredZero()));
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
                           const CuParameter msat2,
                           Grid mastergrid,
                           const CuParameter afmex_nn,
                           const CuParameter aex,
                           const CuParameter aex2,
                           const CuParameter latcon,
                           const bool enableOpenBC,
			                     const int comp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
 real6 m; 
  if (comp == 3) {
	  real3 mag = mField.FM_vectorAt(idx);
	  m = real6{mag.x, mag.y, mag.z, 0, 0, 0};
  }
  else if (comp == 6) 
	  m = mField.AFM_vectorAt(idx);
  
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0 && msat2.valueAt(idx) == 0)) {
    if (comp == 3)
      hField.setVectorInCell(idx, real3{0, 0, 0});
    else if (comp == 6)
      hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
    return;
  }

  const int3 coo = system.grid.index2coord(idx);

  // Accumulate DMI field of cell at idx in h. Divide by msat at the end.
  // Treat 6d case and strip 3 zeros at the end in case of FM.
  real6 h{0, 0, 0, 0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx = system.grid.coord2index(neighbor_coo);

    // If we assume open boundary conditions and if there is no neighbor, 
    // then simply continue without adding to the effective field.    
    if ((!system.inGeometry(neighbor_coo) && enableOpenBC)
       || (msat.valueAt(neighbor_idx) == 0 && msat2.valueAt(neighbor_idx) == 0))
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

    real6 m_;

    if (!system.inGeometry(neighbor_coo)) {
    // DMI-BC
      real a = aex.valueAt(idx);
      if (comp == 6)
        a = harmonicMean(aex.valueAt(idx), aex2.valueAt(idx));

      real fac = afmex_nn.valueAt(idx) / (2 * a);
      real Ax = Dxz * system.cellsize.x / (2 * a * (1 - fac * fac));
      real Ay = Dyz * system.cellsize.y / (2 * a * (1 - fac * fac));

      if (relative_coo.x) {
        m_.x1 = m.x1 - relative_coo.x * Ax * (m.z1 - fac * m.z2);
        m_.y1 = m.y1;
        m_.z1 = m.z1 + relative_coo.x * Ax * (m.x1 - fac * m.x2);
        m_.x2 = m.x2 - relative_coo.x * Ax * (m.z2 - fac * m.z1);
        m_.y2 = m.y2;
        m_.z2 = m.z2 + relative_coo.x * Ax * (m.x2 - fac * m.x1);
      }
      else if (relative_coo.y) {
        m_.x1 = m.x1;
        m_.y1 = m.y1 - relative_coo.y * Ay * (m.z1 - fac * m.z2);
        m_.z1 = m.z1 + relative_coo.y * Ay * (m.y1 - fac * m.y2);
        m_.x2 = m.x2;
        m_.y2 = m.y2 - relative_coo.y * Ay * (m.z2 - fac * m.z1);
        m_.z2 = m.z2 + relative_coo.y * Ay * (m.y2 - fac * m.y1);
      }
      else if (relative_coo.z) {
        m_ = m;
      }
    }
    else {
      if (comp == 3) {
        real3 mag_ = mField.FM_vectorAt(neighbor_idx);
        m_ = real6{mag_.x, mag_.y, mag_.z, 0, 0, 0};
      }
      else if (comp == 6)
        m_ = mField.AFM_vectorAt(neighbor_idx);
    }
    // Compute the effective field contribution of the DMI with the neighbor
    h.x1 += (Dxy * m_.y1 + Dxz * m_.z1) / delta;
    h.y1 += (Dyx * m_.x1 + Dyz * m_.z1) / delta;
    h.z1 += (Dzx * m_.x1 + Dzy * m_.y1) / delta;
    h.x2 += (Dxy * m_.y2 + Dxz * m_.z2) / delta;
    h.y2 += (Dyx * m_.x2 + Dyz * m_.z2) / delta;
    h.z2 += (Dzx * m_.x2 + Dzy * m_.y2) / delta;
  }  // end loop over neighbors

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
  }

  if(comp == 3) {
    h /= msat.valueAt(idx);
    hField.setVectorInCell(idx, real3{h.x1, h.y1, h.z1});
  }
  else if (comp == 6) {
    h /= real2{msat.valueAt(idx), msat2.valueAt(idx)};
    hField.setVectorInCell(idx, h);
  }
}

Field evalDmiField(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  Field hField(magnet->system(), comp);
  if (dmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  cudaLaunch(hField.grid().ncells(), k_dmiField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->dmiTensor.cu(),
             magnet->msat.cu(), magnet->msat2.cu(), magnet->world()->mastergrid(),
             magnet->afmex_nn.cu(), magnet->aex.cu(), magnet->aex2.cu(),
             magnet->latcon.cu(), magnet->enableOpenBC, comp);
  return hField;
}

Field evalDmiEnergyDensity(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return Field(magnet->system(), magnet->magnetization()->ncomp() / 3, 0.0);

  return evalEnergyDensity(magnet, evalDmiField(magnet), 0.5);
}

real evalDmiEnergy(const Ferromagnet* magnet, const bool sub2) {
  if (dmiAssuredZero(magnet))
    return 0;

  real edens;
  if (!sub2) 
    edens = dmiEnergyDensityQuantity(magnet).average()[0];
  else 
    edens = dmiEnergyDensityQuantity(magnet).average()[1];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity dmiFieldQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalDmiField, comp, "dmi_field", "T");
}

FM_FieldQuantity dmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalDmiEnergyDensity, comp / 3, "dmi_emergy_density",
                          "J/m3");
}

FM_ScalarQuantity dmiEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  std::string name = sub2 ? "dmi_energy2" : "dmi_energy";
  return FM_ScalarQuantity(magnet, evalDmiEnergy, sub2, name, "J");
}
