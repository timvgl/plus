#include <memory>

#include "antiferromagnet.hpp"
#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "magnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "system.hpp"
#include "totalmag.hpp"

__global__ void k_demagfield(CuField hField,
                             const CuField mField,
                             const CuField kernel,
                             const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry of destiny field, set to zero and return
  // early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  int3 dstcoo = hField.system.grid.index2coord(idx);
  real3 h{0, 0, 0};

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
    
    real3 M = msat.valueAt(i) * mField.FM_vectorAt(i);

    h.x -= nxx * M.x + nxy * M.y + nxz * M.z;
    h.y -= nxy * M.x + nyy * M.y + nyz * M.z;
    h.z -= nxz * M.x + nyz * M.y + nzz * M.z;
  }
  
  hField.setVectorInCell(idx, MU0 * h);
}

StrayFieldBruteExecutor::StrayFieldBruteExecutor(
    const Magnet* magnet,
    std::shared_ptr<const System> system)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(), magnet_->grid(), magnet_->world()) {}

Field StrayFieldBruteExecutor::exec() const {
  
  Field h(system_, 3);
  int ncells = h.grid().ncells();

  if(const Ferromagnet* mag = dynamic_cast<const Ferromagnet*>(magnet_)) {
    auto m = mag->magnetization()->field().cu();
    cudaLaunch(ncells, k_demagfield, h.cu(), m, kernel_.field().cu(),
              mag->msat.cu());
  }
  else if (const Antiferromagnet* mag = dynamic_cast<const Antiferromagnet*>(magnet_)) {
    auto m = add(mag->sub1()->magnetization()->field(), mag->sub2()->magnetization()->field());

    cudaLaunch(ncells, k_demagfield, h.cu(), m.cu(), kernel_.field().cu(),
              mag->sub1()->msat.cu());
  }
  return h;
}
