#include <memory>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "system.hpp"

__global__ void k_demagfield(CuField hField,
                             const CuField mField,
                             const CuField kernel,
                             const CuParameter msat,
                             const CuParameter msat2,
                             bool afm) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry of destiny field, set to zero and return
  // early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx)) {
      if (!afm) {
        hField.setVectorInCell(idx, real3{0, 0, 0});
      }
      else {
        hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
      }
    }
    return;
  }

  // Treat 6d case and strip 3 zeros at the end in case of FM.
  real6 h{0, 0, 0, 0, 0, 0};
  real6 M{0, 0, 0, 0, 0, 0};

  int3 dstcoo = hField.system.grid.index2coord(idx);

  for (int i = 0; i < mField.system.grid.ncells(); i++) {
    if (!mField.cellInGeometry(i))
      continue;

    int3 srccoo = mField.system.grid.index2coord(i);
    int3 r = dstcoo - srccoo;
    real nxx = kernel.valueAt(r, 0);
    real nyy = kernel.valueAt(r, 1);
    real nzz = kernel.valueAt(r, 2);
    real nxy = kernel.valueAt(r, 3);
    real nxz = kernel.valueAt(r, 4);
    real nyz = kernel.valueAt(r, 5);
    
    if (!afm) {
      real3 mag = msat.valueAt(i) * mField.FM_vectorAt(i);
      M = {mag.x, mag.y, mag.z, 0, 0, 0};
    }
    else {
      M = real2{msat.valueAt(i), msat2.valueAt(i)} * mField.AFM_vectorAt(i);
    }

    h.x1 -= nxx * M.x1 + nxy * M.y1 + nxz * M.z1;
    h.y1 -= nxy * M.x1 + nyy * M.y1 + nyz * M.z1;
    h.z1 -= nxz * M.x1 + nyz * M.y1 + nzz * M.z1;
    h.x2 -= nxx * M.x2 + nxy * M.y2 + nxz * M.z2;
    h.y2 -= nxy * M.x2 + nyy * M.y2 + nyz * M.z2;
    h.z2 -= nxz * M.x2 + nyz * M.y2 + nzz * M.z2;
  }
  if (!afm) {
    hField.setVectorInCell(idx, MU0 * real3{h.x1, h.y1, h.z1});
  }
  else {
    hField.setVectorInCell(idx, MU0 * h);
  }
}

StrayFieldBruteExecutor::StrayFieldBruteExecutor(
    const Ferromagnet* magnet,
    std::shared_ptr<const System> system)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(), magnet_->grid(), magnet_->world()) {}

Field StrayFieldBruteExecutor::exec() const {
   auto m = magnet_->magnetization()->field().cu();
  Field h(system_, m.ncomp);
  int ncells = h.grid().ncells();
  bool afm = m.ncomp == 6;
  cudaLaunch(ncells, k_demagfield, h.cu(), m, kernel_.field().cu(),
             magnet_->msat.cu(), magnet_->msat2.cu(), afm);
  return h;
}
