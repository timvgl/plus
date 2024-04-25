#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "neel.hpp"
#include "world.hpp"

__global__ void k_neelvector(CuField neel,
                             const CuField mag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!neel.cellInGeometry(idx)) {
    if (neel.cellInGrid(idx)) 
        neel.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
    real6 m = mag.AFM_vectorAt(idx);
    neel.setVectorInCell(idx, 0.5 * real3{m.x1 - m.x2, m.y1 - m.y2, m.z1 - m.z2}); 
}

Field evalNeelvector(const Ferromagnet* magnet) {
  Field neel(magnet->system(), 3);

  if (magnet->msat.assuredZero() && magnet->msat2.assuredZero()) {
    neel.makeZero();
    return neel;
  }
  cudaLaunch(neel.grid().ncells(), k_neelvector, neel.cu(),
             magnet->magnetization()->field().cu());
  return neel;
}

FM_FieldQuantity neelVectorQuantity(const Ferromagnet* magnet) {
    if (magnet->magnetization()->ncomp() != 6)
        throw std::runtime_error("Cannot compute the Neel vector for a magnetization with"
                                  + std::to_string(magnet->magnetization()->ncomp()) + "components.");
    return FM_FieldQuantity(magnet, evalNeelvector, 3, "neel_vector", "A/m");
}
