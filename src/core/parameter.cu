#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"
#include "reduce.hpp"

namespace {
  struct ParamCleanup {
    Field* staticField = nullptr;
    Field* dynamicField = nullptr;
    cudaEvent_t lastUseEvent = nullptr;
  };

  static void CUDART_CB param_cleanup_cb(void* p) noexcept {
    auto* cap = static_cast<ParamCleanup*>(p);
    if (cap->lastUseEvent) {
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap->staticField;
    delete cap->dynamicField;
    delete cap;
  }
}

Parameter::Parameter(std::shared_ptr<const System> system, real value,
                     std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit) {}

Parameter::Parameter(std::shared_ptr<const System> system, cudaStream_t s, real value,
                     std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit), stream_(s) {}

/* Parameter::~Parameter() {
  waitForLastUse_();
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
  // DynamicParameter<real>::dynamicField_ ist (vermutlich) unique_ptr im Basistyp:
  // sie wird beim Zerstören von 'this' automatisch freigegeben; das Event-Fence oben schützt vorher.
} */

Parameter::~Parameter() {
  scheduleGC_();
}

void Parameter::scheduleGC_() const {
  // nichts zu tun?
  if (!staticField_ && !dynamicField_) {
    if (lastUseEvent_) {
      cudaEventDestroy(lastUseEvent_);
      const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
    }
    return;
  }

  auto* cap = new (std::nothrow) ParamCleanup{};
  if (!cap) {
    // Fallback: blockierend und sicher
    if (lastUseEvent_) {
      cudaEventSynchronize(lastUseEvent_);
      cudaEventDestroy(lastUseEvent_);
      const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
    }
    delete staticField_;
    const_cast<Field*&>(staticField_) = nullptr;
    // dynamicField_ ist unique_ptr im Basistyp – blockierend resetten:
    const_cast<std::unique_ptr<Field>&>(dynamicField_).reset();
    return;
  }

  cap->lastUseEvent = lastUseEvent_;
  const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;

  cap->staticField = staticField_;
  const_cast<Field*&>(staticField_) = nullptr;

  // Basistyp DynamicParameter<real> hält dynamicField_ (unique_ptr)
  cap->dynamicField = const_cast<std::unique_ptr<Field>&>(dynamicField_).release();

  cudaStream_t s_gc = getCudaStreamGC();
  if (cap->lastUseEvent) {
    checkCudaError(cudaStreamWaitEvent(s_gc, cap->lastUseEvent, 0));
  }
  cudaError_t st = cudaLaunchHostFunc(s_gc, param_cleanup_cb, cap);
  if (st != cudaSuccess) {
    // Fallback falls Enqueue fehlschlägt
    if (cap->lastUseEvent) {
      cudaEventSynchronize(cap->lastUseEvent);
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap->staticField;
    delete cap->dynamicField;
    delete cap;
  }
}

void Parameter::markLastUse() const {
  cudaStream_t s = nullptr;
  if (staticField_) s = staticField_->getStream();
  if (!s) s = getCudaStreamGC();
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}

void Parameter::markLastUse(cudaStream_t s) const {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}

/* void Parameter::set(real value) {
  waitForLastUse_();         
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
} */

void Parameter::set(real value) {
  scheduleGC_();  // statt waitForLastUse_ + delete
  uniformValue_ = value;
}

/* void Parameter::set(const Field& values) {
  waitForLastUse_(); 
  if (isUniformField(values)) {
    real* value = values.device_ptr(0);
    checkCudaError(cudaMemcpy(&uniformValue_, value, sizeof(real),
                            cudaMemcpyDeviceToHost));
    if (staticField_) {
      delete staticField_;
      staticField_ = nullptr;
    }
  }
  else
    staticField_ = new Field(values);
} */

void Parameter::set(const Field& values) {
  scheduleGC_();
  if (isUniformField(values)) {
    real* value = values.device_ptr(0);
    checkCudaError(cudaMemcpy(&uniformValue_, value, sizeof(real),
                              cudaMemcpyDeviceToHost));
  } else {
    staticField_ = new Field(values);
  }
}

/* void Parameter::setInRegion(const unsigned int region_idx, real value) {
  waitForLastUse_();
  if (isUniform()) {
    if (value == uniformValue_) return;
    staticField_ = new Field(system_, 1, uniformValue_);
  }
  staticField_->setUniformValueInRegion(region_idx, value);
} */

void Parameter::setInRegion(const unsigned int region_idx, real value) {
  // Falls bislang uniform: statisches Feld anlegen (neu), vorheriges per GC räumen
  if (isUniform()) {
    if (value == uniformValue_) return;
    scheduleGC_();
    staticField_ = new Field(system_, 1, uniformValue_);
  }
  staticField_->setUniformValueInRegion(region_idx, value);
}

bool Parameter::isUniform() const {
  return !staticField_ && DynamicParameter<real>::isUniform();
}

bool Parameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

std::shared_ptr<const System> Parameter::system() const {
  return system_;
}

Field Parameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);
    staticField += dynamicField;
    dynamicField.markLastUse();
  }
  staticField.markLastUse();
  return staticField;
}

Field Parameter::eval(cudaStream_t s) const {
  Field staticField(system_, ncomp(), s);

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_, s);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp(), s);

    evalTimeDependentTerms(t, dynamicField, s);
    addTo(staticField, real{1}, dynamicField, s);
    dynamicField.markLastUse(s);
  }
  staticField.markLastUse(s);
  return staticField;
}

real Parameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument("Cannot get uniform value of non-uniform Parameter.");
  }
  return uniformValue_;
}

CuParameter Parameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
    dynamicField_->markLastUse();
  }

  return CuParameter(this);
}

CuParameter Parameter::cu(cudaStream_t s) const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp(), s));

    evalTimeDependentTerms(t, *dynamicField_, s);
    dynamicField_->markLastUse(s);
  }

  if (staticField_) {
    fenceStreamToStream(staticField_->getStream(), s);
  }

  return CuParameter(this);
}

VectorParameter::VectorParameter(std::shared_ptr<const System> system,
                                 real3 value,
                                 std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit) {}

/* VectorParameter::~VectorParameter() {
  waitForLastUse_();
  if (staticField_)
    delete staticField_;
    staticField_ = nullptr;
} */

VectorParameter::~VectorParameter() {
  scheduleGC_();
}

void VectorParameter::scheduleGC_() const {
  if (!staticField_ && !dynamicField_) {
    if (lastUseEvent_) {
      cudaEventDestroy(lastUseEvent_);
      const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
    }
    return;
  }

  auto* cap = new (std::nothrow) ParamCleanup{};
  if (!cap) {
    if (lastUseEvent_) {
      cudaEventSynchronize(lastUseEvent_);
      cudaEventDestroy(lastUseEvent_);
      const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
    }
    delete staticField_;
    const_cast<Field*&>(staticField_) = nullptr;
    const_cast<std::unique_ptr<Field>&>(dynamicField_).reset();
    return;
  }

  cap->lastUseEvent = lastUseEvent_;
  const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;

  cap->staticField = staticField_;
  const_cast<Field*&>(staticField_) = nullptr;

  cap->dynamicField = const_cast<std::unique_ptr<Field>&>(dynamicField_).release();

  cudaStream_t s_gc = getCudaStreamGC();
  if (cap->lastUseEvent) {
    checkCudaError(cudaStreamWaitEvent(s_gc, cap->lastUseEvent, 0));
  }
  cudaError_t st = cudaLaunchHostFunc(s_gc, param_cleanup_cb, cap);
  if (st != cudaSuccess) {
    if (cap->lastUseEvent) {
      cudaEventSynchronize(cap->lastUseEvent);
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap->staticField;
    delete cap->dynamicField;
    delete cap;
  }
}

void VectorParameter::markLastUse() const {
  cudaStream_t s = nullptr;
  if (staticField_) s = staticField_->getStream();
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}

void VectorParameter::markLastUse(cudaStream_t s) const {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}

/* void VectorParameter::set(real3 value) {
  waitForLastUse_();
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
} */

void VectorParameter::set(real3 value) {
  scheduleGC_();
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

/* void VectorParameter::set(const Field& values) {
  waitForLastUse_();
  if (isUniformField(values)) {
    real* valueX = values.device_ptr(0);
    real* valueY = values.device_ptr(1);
    real* valueZ = values.device_ptr(2);

    checkCudaError(cudaMemcpy(&uniformValue_.x, valueX, sizeof(real),
                            cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(&uniformValue_.y, valueY, sizeof(real),
                            cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(&uniformValue_.z, valueZ, sizeof(real),
                            cudaMemcpyDeviceToHost));
    if (staticField_) {
      delete staticField_;
      staticField_ = nullptr;
    }
  }
  else
    staticField_ = new Field(values);
} */

void VectorParameter::set(const Field& values) {
  scheduleGC_();
  if (isUniformField(values)) {
    real* valueX = values.device_ptr(0);
    real* valueY = values.device_ptr(1);
    real* valueZ = values.device_ptr(2);

    checkCudaError(cudaMemcpy(&uniformValue_.x, valueX, sizeof(real),
                            cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(&uniformValue_.y, valueY, sizeof(real),
                            cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(&uniformValue_.z, valueZ, sizeof(real),
                            cudaMemcpyDeviceToHost));
    if (staticField_) {
      delete staticField_;
      staticField_ = nullptr;
    }
  }
  else
    staticField_ = new Field(values);
}

/* void VectorParameter::setInRegion(const unsigned int region_idx, real3 value) {
  waitForLastUse_();
  if (isUniform()) {
    if (value == uniformValue_) return;
    staticField_ = new Field(system_, 3, uniformValue_);
  }
  staticField_->setUniformValueInRegion(region_idx, value);
} */

void VectorParameter::setInRegion(const unsigned int region_idx, real3 value) {
  // Falls bislang uniform: statisches Feld anlegen (neu), vorheriges per GC räumen
  if (isUniform()) {
    if (value == uniformValue_) return;
    scheduleGC_();
    staticField_ = new Field(system_, 3, uniformValue_);
  }
  staticField_->setUniformValueInRegion(region_idx, value);
}

bool VectorParameter::isUniform() const {
  return !staticField_ && DynamicParameter<real3>::isUniform();
}

bool VectorParameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

std::shared_ptr<const System> VectorParameter::system() const {
  return system_;
}

Field VectorParameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
    dynamicField.markLastUse();
  }
  staticField.markLastUse();
  return staticField;
}

real3 VectorParameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument("Cannot get uniform value of non-uniform Parameter.");
  }
  return uniformValue_;
}

CuVectorParameter VectorParameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
    dynamicField_->markLastUse();
  }

  return CuVectorParameter(this);
}
