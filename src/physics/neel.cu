#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "neel.hpp"

__global__ void k_neelvector(CuField neel,
                             const CuField mag1,
                             const CuField mag2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!neel.cellInGeometry(idx)) {
    if (neel.cellInGrid(idx)) 
        neel.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
    real3 m1 = mag1.FM_vectorAt(idx);
    real3 m2 = mag2.FM_vectorAt(idx);
    neel.setVectorInCell(idx, 0.5 * (m1 - m2));
}

Field evalNeelvector(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  Field neel(magnet->system(), 3);

  if (magnet->sub1()->msat.assuredZero() && magnet->sub2()->msat.assuredZero()) {
    neel.makeZero();
    return neel;
  }
  cudaLaunch(neel.grid().ncells(), k_neelvector, neel.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu());
  return neel;
}

AFM_FieldQuantity neelVectorQuantity(const Antiferromagnet* magnet) {
    // TODO: make sublattice argument optional (nullptr)
    return AFM_FieldQuantity(magnet, magnet->sub1(), evalNeelvector, 3, "neel_vector", "A/m");
}
