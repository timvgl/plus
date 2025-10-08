#pragma once

#include <functional>
#include <memory>
#include <utility>

#include "datatypes.hpp"
#include "dynamic_parameter.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "grid.hpp"
#include "system.hpp"

class CuParameter;

class Parameter : public FieldQuantity, public DynamicParameter<real> {
 public:
  explicit Parameter(std::shared_ptr<const System> system, real value = 0.0,
                     std::string name = "", std::string unit = "");
  explicit Parameter(std::shared_ptr<const System> system,  cudaStream_t s, real value = 0.0,
                     std::string name = "", std::string unit = "");
  ~Parameter();
  
  void markLastUse() const;
  void markLastUse(cudaStream_t s) const;

  void set(real value);
  void set(const Field& values);
  void setInRegion(const unsigned int region_idx, real value);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  std::string name() const {return name_;}
  std::string unit() const {return unit_;}
  /** Evaluate parameter on its field. */
  Field eval() const;
  Field eval(cudaStream_t stream_) const;
  /** Get value of uniform parameter.*/
  real getUniformValue() const;

  /** Send parameter data to the device. */
  CuParameter cu() const;
  CuParameter cu(cudaStream_t s) const;

 private:
  std::shared_ptr<const System> system_;
  real uniformValue_;
  /** Store time-independent values. */
  Field* staticField_;

  std::string name_;
  std::string unit_;
  mutable cudaEvent_t lastUseEvent_ = nullptr;
  cudaStream_t stream_ = nullptr;  // non-owning, borrowed
  void scheduleGC_() const;

  friend CuParameter;
};

class CuParameter {
 public:
  const CuSystem system;
  const real uniformValue;

 private:
  real* valuesPtr;
  real* dynamicValuesPtr;

 public:
  explicit CuParameter(const Parameter* p);
  __device__ bool isUniform() const;
  __device__ real valueAt(int idx) const;
  __device__ real valueAt(int3 coo) const;
  __device__ inline real harmonicMean(int idx1, int idx2) const;
};

inline CuParameter::CuParameter(const Parameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      valuesPtr(nullptr),
      dynamicValuesPtr(nullptr) {
  if (p->staticField_) {
    valuesPtr = p->staticField_->device_ptr(0);
  }

  if (p->dynamicField_) {
    dynamicValuesPtr = p->dynamicField_->device_ptr(0);
  }
}

__device__ inline bool CuParameter::isUniform() const {
  return !valuesPtr;
}

__device__ inline real CuParameter::valueAt(int idx) const {
  if (isUniform()) {
    if (dynamicValuesPtr) {
      return uniformValue + dynamicValuesPtr[idx];
    }

    return uniformValue;
  } else {
    if (dynamicValuesPtr) {
      return valuesPtr[idx] + dynamicValuesPtr[idx];
    }

    return valuesPtr[idx];
  }
}

__device__ inline real CuParameter::valueAt(int3 coo) const {
  return valueAt(system.grid.coord2index(coo));
}

__device__ inline real CuParameter::harmonicMean(int idx1, int idx2) const {
  if (idx1 == idx2) { return valueAt(idx1); }
  else { return harmonicMean(valueAt(idx1), valueAt(idx2)); }
}


class CuVectorParameter;

class VectorParameter : public FieldQuantity, public DynamicParameter<real3> {
 public:
  VectorParameter(std::shared_ptr<const System> system,
                  real3 value = {0.0, 0.0, 0.0},
                  std::string name = "", std::string unit = "");
  ~VectorParameter();

  void markLastUse() const;
  void markLastUse(cudaStream_t s) const;

  void set(real3 value);
  void set(const Field& values);
  void setInRegion(const unsigned int region_idx, real3 value);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  std::shared_ptr<const System> system() const;
  std::string name() const {return name_;}
  std::string unit() const {return unit_;}
  Field eval() const;
  Field eval(cudaStream_t stream_) const;
  real3 getUniformValue() const;

  CuVectorParameter cu() const;

 private:
  std::shared_ptr<const System> system_;
  real3 uniformValue_;
  Field* staticField_;

  std::string name_;
  std::string unit_;
  mutable cudaEvent_t lastUseEvent_ = nullptr;
  cudaStream_t stream_ = nullptr;  // non-owning, borrowed
  void scheduleGC_() const;

  friend CuVectorParameter;
};

struct CuVectorParameter {
 public:
  const CuSystem system;
  const real3 uniformValue;

 private:
  real* xValuesPtr;
  real* yValuesPtr;
  real* zValuesPtr;
  real* xDynamicValuesPtr;
  real* yDynamicValuesPtr;
  real* zDynamicValuesPtr;

 public:
  explicit CuVectorParameter(const VectorParameter*);
  __device__ bool isUniform() const;

  __device__ real valueAt(int idx, int comp = 0) const;
  __device__ real valueAt(int3 coo, int comp = 0) const;

  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;
};

inline CuVectorParameter::CuVectorParameter(const VectorParameter* p)
    : system(p->system()->cu()),
      uniformValue(p->uniformValue_),
      xValuesPtr(nullptr),
      yValuesPtr(nullptr),
      zValuesPtr(nullptr),
      xDynamicValuesPtr(nullptr),
      yDynamicValuesPtr(nullptr),
      zDynamicValuesPtr(nullptr) {
  if (p->staticField_) {
    xValuesPtr = p->staticField_->device_ptr(0);
    yValuesPtr = p->staticField_->device_ptr(1);
    zValuesPtr = p->staticField_->device_ptr(2);
  }

  if (p->dynamicField_) {
    xDynamicValuesPtr = p->dynamicField_->device_ptr(0);
    yDynamicValuesPtr = p->dynamicField_->device_ptr(1);
    zDynamicValuesPtr = p->dynamicField_->device_ptr(2);
  }
}

__device__ inline bool CuVectorParameter::isUniform() const {
  return !xValuesPtr;
}

__device__ inline real CuVectorParameter::valueAt(int idx, int comp) const {
  real sv, dv;  // static and dynamic values
  if (comp == 0) {  // x
    sv = isUniform() ? uniformValue.x : xValuesPtr[idx];
    dv = xDynamicValuesPtr ? xDynamicValuesPtr[idx] : 0.;
  } else if (comp == 1) {  // y
    sv = isUniform() ? uniformValue.y : yValuesPtr[idx];
    dv = yDynamicValuesPtr ? yDynamicValuesPtr[idx] : 0.;
  } else {  // z  comp == 2 (no safety checks)
    sv = isUniform() ? uniformValue.z : zValuesPtr[idx];
    dv = zDynamicValuesPtr ? zDynamicValuesPtr[idx] : 0.;
  }
  return sv + dv;
}

__device__ inline real CuVectorParameter::valueAt(int3 coo, int comp) const {
  return valueAt(system.grid.coord2index(coo), comp);
}

__device__ inline real3 CuVectorParameter::vectorAt(int idx) const {
  if (isUniform()) {
    if (xDynamicValuesPtr && yDynamicValuesPtr && zDynamicValuesPtr) {
      real3 dynamic_value{xDynamicValuesPtr[idx], yDynamicValuesPtr[idx], zDynamicValuesPtr[idx]};

      return uniformValue + dynamic_value;
    }

    return uniformValue;
  } else {
    real3 static_value{xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};
    if (xDynamicValuesPtr && yDynamicValuesPtr && zDynamicValuesPtr) {
      real3 dynamic_value{xDynamicValuesPtr[idx], yDynamicValuesPtr[idx], zDynamicValuesPtr[idx]};

      return static_value + dynamic_value;
    }

    return static_value;
  }
}

__device__ inline real3 CuVectorParameter::vectorAt(int3 coo) const {
  return vectorAt(system.grid.coord2index(coo));
}
