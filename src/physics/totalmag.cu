#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "totalmag.hpp"

__global__ void k_totalmag(CuField total,
                             const CuField mag1,
                             const CuField mag2,
                             const CuParameter msat1,
                             const CuParameter msat2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!total.cellInGeometry(idx)) {
    if (total.cellInGrid(idx)) 
        total.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
    real3 m1 = mag1.vectorAt(idx);
    real3 m2 = mag2.vectorAt(idx);
    real ms1 = msat1.valueAt(idx);
    real ms2 = msat2.valueAt(idx);

    total.setVectorInCell(idx, ms1 * m1 + ms2 * m2);
}

Field evalTotalMag(const Antiferromagnet* magnet) {
  Field total(magnet->system(), 3);

  if (magnet->sub1()->msat.assuredZero() && magnet->sub2()->msat.assuredZero()) {
    total.makeZero();
    return total;
  }
  cudaLaunch(total.grid().ncells(), k_totalmag, total.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub1()->msat.cu(), magnet->sub2()->msat.cu());
  return total;
}

AFM_FieldQuantity totalMagnetizationQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalTotalMag, 3, "total_magnetization", "A/m");
}
