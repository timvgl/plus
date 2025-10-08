#include "dmitensor.hpp"

#include <memory>
#include <vector>

#include "parameter.hpp"
#include "system.hpp"

DmiTensor::DmiTensor(std::shared_ptr<const System> system)
    : xxy(system),
      xyz(system),
      xxz(system),
      yxy(system),
      yyz(system),
      yxz(system),
      zxy(system),
      zyz(system),
      zxz(system) {}

CuDmiTensor DmiTensor::cu() const {
  return CuDmiTensor{
      xxy.cu(), xyz.cu(), xxz.cu(), yxy.cu(), yyz.cu(),
      yxz.cu(), zxy.cu(), zyz.cu(), zxz.cu(),
  };
}

CuDmiTensor DmiTensor::cu(cudaStream_t s) const {
  return CuDmiTensor{
    xxy.cu(s), xyz.cu(s), xxz.cu(s),
    yxy.cu(s), yyz.cu(s), yxz.cu(s),
    zxy.cu(s), zyz.cu(s), zxz.cu(s)
  };
}

void DmiTensor::markLastUse() const {                     
  xxy.markLastUse();  xyz.markLastUse();  xxz.markLastUse();
  yxy.markLastUse();  yyz.markLastUse();  yxz.markLastUse();
  zxy.markLastUse();  zyz.markLastUse();  zxz.markLastUse();
}

void DmiTensor::markLastUse(cudaStream_t s) const {    
  xxy.markLastUse(s);  xyz.markLastUse(s);  xxz.markLastUse(s);
  yxy.markLastUse(s);  yyz.markLastUse(s);  yxz.markLastUse(s);
  zxy.markLastUse(s);  zyz.markLastUse(s);  zxz.markLastUse(s);
}

bool DmiTensor::assuredZero() const {
  return xxy.assuredZero() && xyz.assuredZero() && xxz.assuredZero() &&
         yxy.assuredZero() && yyz.assuredZero() && yxz.assuredZero() &&
         zxy.assuredZero() && zyz.assuredZero() && zxz.assuredZero();
}