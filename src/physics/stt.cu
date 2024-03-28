#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "world.hpp"

bool spinTransferTorqueAssuredZero(const Ferromagnet* magnet) {
  return (magnet->msat.assuredZero() && magnet->msat2.assuredZero()) || magnet->jcur.assuredZero() ||
         magnet->pol.assuredZero();
}

bool ZhangLiSTTAssuredZero(const Ferromagnet* magnet) {
  return spinTransferTorqueAssuredZero(magnet) || (!magnet->Lambda.assuredZero());
}

bool SlonczewskiSTTAssuredZero(const Ferromagnet* magnet) {
  return ZhangLiSTTAssuredZero(magnet) || magnet->Lambda.assuredZero();
}
__global__ void k_ZhangLi(CuField torque,
                                     const CuField mField,
                                     const CuParameter msatParam,
                                     const CuParameter msat2Param,
                                     const CuParameter polParam,
                                     const CuParameter xiParam,
                                     const CuParameter alphaParam,
                                     const FM_CuVectorParameter jcurParam,
                                     const int comp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const Grid grid = torque.system.grid;
  const real3 cellsize = torque.system.cellsize;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      if (comp == 3)
        torque.setVectorInCell(idx, real3{0, 0, 0});
      else if (comp == 6)
        torque.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
    return;
  }

  // Treat the 6d case and strip 3 zeros at the end in case of FM
  real6 m;
  if (comp == 3) {
    const real3 mag = mField.FM_vectorAt(idx);
    m = {mag.x, mag.y, mag.z, 0, 0, 0};
  }
  else {
    m = mField.AFM_vectorAt(idx);
  }
  const real3 jj = jcurParam.FM_vectorAt(idx);
  const real6 j = {jj.x, jj.y, jj.z, jj.x, jj.y, jj.z};
  const real msat = msatParam.valueAt(idx);
  const real msat2 = msat2Param.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real xi = xiParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);

  if ((msat == 0 && msat2 == 0) || pol == 0 || j == real6{0, 0, 0, 0, 0, 0}) {
    if (comp == 3)
      torque.setVectorInCell(idx, real3{0, 0, 0});
    else
      torque.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);

  const real6 u = MUB * pol / (QE * real2{msat, msat2} * (1 + xi * xi)) * j;

  real6 hspin{0, 0, 0, 0, 0, 0};

  // x derivative
  for (int sign : {-1, 1}) {  // left and right neighbor
    const int3 coo_ = coo + int3{sign, 0, 0};
    if (grid.cellInGrid(coo_) && (msatParam.valueAt(coo_) != 0 || msat2Param.valueAt(coo_) != 0)) {
      real6 m_;
      if (comp == 3) {
        real3 mag_ = mField.FM_vectorAt(coo_);
        m_ = real6{mag_.x, mag_.y, mag_.z, 0, 0, 0};
      }
      else if (comp == 6) {
        m_ = mField.AFM_vectorAt(coo_);
      }
      hspin += sign * real2{u.x1, u.x2} * m_ / (2 * cellsize.x);  // central finite difference
    }
  }
  // y derivative
  for (int sign : {-1, 1}) {
    const int3 coo_ = coo + int3{0, sign, 0};
    if (grid.cellInGrid(coo_) && (msatParam.valueAt(coo_) != 0 || msat2Param.valueAt(coo_) != 0)) {
      real6 m_;
      if (comp == 3) {
        real3 mag_ = mField.FM_vectorAt(coo_);
        m_ = real6{mag_.x, mag_.y, mag_.z, 0, 0, 0};
      }
      else if (comp == 6) {
        m_ = mField.AFM_vectorAt(coo_);
      }
      hspin += sign * real2{u.y1, u.y2} * m_ / (2 * cellsize.y);
    }
  }
  // z derivative
  for (int sign : {-1, 1}) {
    const int3 coo_ = coo + int3{0, 0, sign};
    if (grid.cellInGrid(coo_) && (msatParam.valueAt(coo_) != 0 || msatParam.valueAt(coo_) != 0)) {
      real6 m_;
      if (comp == 3) {
        real3 mag_ = mField.FM_vectorAt(coo_);
        m_ = real6{mag_.x, mag_.y, mag_.z, 0, 0, 0};
      }
      else if (comp == 6) {
        m_ = mField.AFM_vectorAt(coo_);
      }
      hspin += sign * real2{u.z1, u.z2} * m_ / (2 * cellsize.z);  // central finite difference
    }
  }

  const real6 mxh = cross(m, hspin);
  const real6 mxmxh = cross(m, mxh);
  const real6 t = (-1 / (1 + alpha * alpha)) *
                  ((1 + xi * alpha) * mxmxh + (xi - alpha) * mxh); // In 1/s
  if (comp == 3)
    torque.setVectorInCell(idx, real3{t.x1, t.y1, t.z1});
  else if (comp == 6)
    torque.setVectorInCell(idx, t);
}

__global__ void k_Slonczewski(CuField torque,
                                     const CuField mField,
                                     const CuParameter msatParam,
                                     const CuParameter msat2Param,
                                     const CuParameter polParam,
                                     const CuParameter lambdaParam,
                                     const CuParameter alphaParam,
                                     const FM_CuVectorParameter jcurParam,
                                     const CuParameter eps_prime,
                                     const FM_CuVectorParameter FixedLayer,
                                     const CuParameter FreeLayerThickness,
                                     const int comp) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      if (comp == 3)
        torque.setVectorInCell(idx, real3{0, 0, 0});
      else if (comp == 6)
        torque.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
    return;
  }

  // Treat the 6d case and strip 3 zeros at the end in case of FM
  real6 m;
  if (comp == 3) {
    const real3 mag = mField.FM_vectorAt(idx);
    m = {mag.x, mag.y, mag.z, 0, 0, 0};
  }
  else {
    m = mField.AFM_vectorAt(idx);
  }

  const real3 jj = jcurParam.FM_vectorAt(idx);
  const real j = norm(jj);

  const real msat = msatParam.valueAt(idx);
  const real msat2 = msat2Param.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);

  const real3 p = FixedLayer.FM_vectorAt(idx);
  const real lambda = lambdaParam.valueAt(idx);
  const real eps_p = eps_prime.valueAt(idx);
  const real d = FreeLayerThickness.valueAt(idx);

  if ((msat == 0 && msat2 == 0) || pol == 0 || jj == real3{0, 0, 0}) {
    if (comp == 3)
      torque.setVectorInCell(idx, real3{0, 0, 0});
    else
      torque.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
    return;
  }

  const real2 B = (HBAR / QE) * j / (real2{msat, msat2} * d); // in Tesla
  const real lambda2 = lambda * lambda;
  const real2 eps = pol * lambda2 / ((lambda2 + 1) + (lambda2 - 1) * dot(m, p));

 
  const real6 mxp = cross(m, p);
  const real6 mxmxp = cross(m, mxp);
  const real6 t = (- (eps  + eps_p * alpha) * mxmxp 
                   - (eps_p - eps  * alpha) * mxp) * (B / (1 + alpha * alpha)) * GAMMALL;

  if (comp == 3)
    torque.setVectorInCell(idx, real3{t.x1, t.y1, t.z1});
  else if (comp == 6)
    torque.setVectorInCell(idx, t);
}

Field evalSpinTransferTorque(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  Field torque(magnet->system(), comp);

  if (spinTransferTorqueAssuredZero(magnet)) {
    torque.makeZero();
    return torque;
  }

  int ncells = magnet->grid().ncells();
  auto m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto msat2 = magnet->msat2.cu();
  auto pol = magnet->pol.cu();
  auto xi = magnet->xi.cu();
  auto alpha = magnet->alpha.cu();
  auto jcur = magnet->jcur.cu();
  auto lambda = magnet->Lambda.cu();
  auto eps_prime = magnet->eps_prime.cu();
  auto FixedLayer = magnet->FixedLayer.cu();
  auto FreeLayerThickness = magnet->FreeLayerThickness.cu();

  auto cellsize = magnet->world()->cellsize();

  if (magnet->Lambda.assuredZero() && magnet->eps_prime.assuredZero())
    cudaLaunch(ncells, k_ZhangLi, torque.cu(), m, msat, msat2, pol, xi, alpha,
             jcur, comp);
  else
    cudaLaunch(ncells, k_Slonczewski, torque.cu(), m, msat, msat2, pol, lambda, alpha,
             jcur, eps_prime, FixedLayer, FreeLayerThickness, comp);
  return torque;
}

FM_FieldQuantity spinTransferTorqueQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalSpinTransferTorque, comp,
                          "spintransfer_torque", "1/s");
}
