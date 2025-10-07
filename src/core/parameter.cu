#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"
#include "reduce.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value,
                     std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit) {}

Parameter::~Parameter() {
  waitForLastUse_();
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
  // DynamicParameter<real>::dynamicField_ ist (vermutlich) unique_ptr im Basistyp:
  // sie wird beim Zerstören von 'this' automatisch freigegeben; das Event-Fence oben schützt vorher.
}

void Parameter::waitForLastUse_() const {
  if (lastUseEvent_) {
    cudaEventSynchronize(lastUseEvent_);
    cudaEventDestroy(lastUseEvent_);
    const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
  }
}

void Parameter::markLastUse() const {
  // Fallback ohne expliziten Stream: versuche, aus statischem Feld den Stream zu nehmen
  cudaStream_t s = nullptr;
  if (staticField_) s = staticField_->getStream();
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

void Parameter::set(real value) {
  waitForLastUse_();         
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

void Parameter::set(const Field& values) {
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
}

void Parameter::setInRegion(const unsigned int region_idx, real value) {
  waitForLastUse_();
  if (isUniform()) {
    if (value == uniformValue_) return;
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
  }

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
  }

  return CuParameter(this);
}

VectorParameter::VectorParameter(std::shared_ptr<const System> system,
                                 real3 value,
                                 std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit) {}

VectorParameter::~VectorParameter() {
  waitForLastUse_();
  if (staticField_)
    delete staticField_;
    staticField_ = nullptr;
}

void VectorParameter::waitForLastUse_() const {
  if (lastUseEvent_) {
    cudaEventSynchronize(lastUseEvent_);
    cudaEventDestroy(lastUseEvent_);
    const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
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

void VectorParameter::set(real3 value) {
  waitForLastUse_();
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

void VectorParameter::set(const Field& values) {
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
}

void VectorParameter::setInRegion(const unsigned int region_idx, real3 value) {
  waitForLastUse_();
  if (isUniform()) {
    if (value == uniformValue_) return;
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
  }

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
  }

  return CuVectorParameter(this);
}
