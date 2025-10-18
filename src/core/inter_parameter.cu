#include "cudalaunch.hpp"
#include "inter_parameter.hpp"
#include "reduce.hpp"

#include <algorithm>

namespace {
  struct InterParamCleanup {
    GpuBuffer<real> buf;
    cudaEvent_t evt = nullptr;
  };

  static void CUDART_CB interparam_cleanup_cb(void* p) noexcept {
    auto* cap = static_cast<InterParamCleanup*>(p);
    if (cap->evt) {
      cudaEventDestroy(cap->evt);
      cap->evt = nullptr;
    }
    delete cap;
  }
}

InterParameter::~InterParameter() {
  scheduleBufGC_();
}


void InterParameter::scheduleBufGC_() const {
  if (valuesBuffer_.size() == 0 && !lastUseEvent_) return;

  auto* cap = new (std::nothrow) InterParamCleanup{};
  if (!cap) {
    if (lastUseEvent_) {
      cudaEventSynchronize(lastUseEvent_);
      cudaEventDestroy(lastUseEvent_);
      const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;
    }
    return;
  }

  cap->buf = std::move(const_cast<GpuBuffer<real>&>(valuesBuffer_));
  cap->evt = lastUseEvent_;
  const_cast<cudaEvent_t&>(lastUseEvent_) = nullptr;

  cudaStream_t s_gc = getCudaStreamGC();
  if (cap->evt) {
    checkCudaError(cudaStreamWaitEvent(s_gc, cap->evt, 0));
  }
  cudaError_t st = cudaLaunchHostFunc(s_gc, interparam_cleanup_cb, cap);
  if (st != cudaSuccess) {
    // Fallback
    if (cap->evt) {
      cudaEventSynchronize(cap->evt);
      cudaEventDestroy(cap->evt);
      cap->evt = nullptr;
    }
    delete cap;
  }
}


InterParameter::InterParameter(std::shared_ptr<const System> system,
                               real value, std::string name, std::string unit)
    : system_(system),
      name_(name),
      unit_(unit),
      uniformValue_(value),
      valuesBuffer_(),
      stream_(getCudaStream()) {
  size_t N = 1;  // at least 1 region: default 0
  std::vector<unsigned int> uni = system->uniqueRegions;
  if (!uni.empty()) {
    N = *std::max_element(uni.begin(), uni.end()) + 1;
  }
  valuesLimit_ = N * (N - 1) / 2;
}

void InterParameter::markLastUse() const {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  // Wenn wir keinen besseren Stream kennen: den Param-Standardstream
  checkCudaError(cudaEventRecord(lastUseEvent_, stream_ ? stream_ : getCudaStream()));
}

void InterParameter::markLastUse(cudaStream_t s) const {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}


const std::vector<real> InterParameter::eval() const {
  if (isUniform())
    return std::vector<real>(valuesLimit_, uniformValue_);
  return valuesBuffer_.getData();
}

__global__ void k_set(real* values, real value, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  values[idx] = value;
}

void InterParameter::setBuffer(real value) {
  cudaLaunch("inter_parameter.cu", valuesLimit_, k_set, valuesBuffer_.get(), value, static_cast<int>(valuesLimit_));
  //checkCudaError(cudaDeviceSynchronize());
}

void InterParameter::set(real value) {
  scheduleBufGC_();
  uniformValue_ = value;
}

/* void InterParameter::setBetween(unsigned int i, unsigned int j, real value) {
  system_->checkIdxInRegions(i);
  system_->checkIdxInRegions(j);
  if (i == j) {
    throw std::invalid_argument("Can not set " + name_
            + ", region indexes must be different: " + std::to_string(i) + ".");
  }

  if (isUniform()) {
    if (value == uniformValue_) return; // nothing to update

    valuesBuffer_.allocate(valuesLimit_);  // make memory available
    setBuffer(uniformValue_);  // keep old uniform value
  }

  // Update GPU memory directly at (i, j)
  checkCudaError(cudaMemcpy(&valuesBuffer_.get()[getLutIndex(i, j)], &value,
                 sizeof(real), cudaMemcpyHostToDevice));
  return;
} */

/* void InterParameter::setBetween(unsigned int i, unsigned int j, real value) {
  // Falls uniform → neuen Buffer anlegen und mit uniformValue_ füllen
  if (isUniform()) {
    // alter (leer) Buffer hat nichts zu GCen, aber falls ein Event gesetzt war, einfach löschen
    if (lastUseEvent_) {
      cudaEventDestroy(lastUseEvent_);
      lastUseEvent_ = nullptr;
    }
    valuesBuffer_ = GpuBuffer<real>(valuesLimit_, stream_);
    checkCudaError(cudaMemsetAsync(valuesBuffer_.get(), 0, 0, stream_));

    std::vector<real> tmp(valuesLimit_, uniformValue_);
    checkCudaError(cudaMemcpyAsync(valuesBuffer_.get(), tmp.data(),
                                   valuesLimit_*sizeof(real),
                                   cudaMemcpyHostToDevice, stream_));
  } else {
    if (lastUseEvent_) {
      GpuBuffer<real> fresh(valuesLimit_, stream_);
      checkCudaError(cudaMemcpyAsync(fresh.get(), valuesBuffer_.get(),
                                     valuesLimit_*sizeof(real),
                                     cudaMemcpyDeviceToDevice, stream_));
      
      scheduleBufGC_();
      valuesBuffer_ = std::move(fresh);
    }
  }

  // Jetzt ist valuesBuffer_ exklusiv und darf geschrieben werden.
  int idx = getLutIndex(i, j);
  checkCudaError(cudaMemcpyAsync(valuesBuffer_.get() + idx, &value, sizeof(real),
                                 cudaMemcpyHostToDevice, stream_));
}
 */

void InterParameter::setBetween(unsigned int i, unsigned int j, real value) {
  system_->checkIdxInRegions(i);
  system_->checkIdxInRegions(j);
  if (i == j) {
    throw std::invalid_argument("Can not set " + name_
            + ", region indexes must be different: " + std::to_string(i) + ".");
  }
  const size_t bytes = valuesLimit_ * sizeof(real);

  if (isUniform()) {
    // Event wegräumen
    if (lastUseEvent_) { cudaEventDestroy(lastUseEvent_); lastUseEvent_ = nullptr; }
    // neuen aktiven Buffer anlegen & füllen
    valuesBuffer_ = GpuBuffer<real>(valuesLimit_, stream_);
    std::vector<real> tmp(valuesLimit_, uniformValue_);
    checkCudaError(cudaMemcpyAsync(valuesBuffer_.get(), tmp.data(), bytes,
                                   cudaMemcpyHostToDevice, stream_));
  } else {
    // 1) alte Nutzer fertig?
    if (lastUseEvent_) {
      checkCudaError(cudaStreamWaitEvent(stream_, lastUseEvent_, 0));
    }

    // 2) alten Buffer herauslösen + fresh anlegen + D2D-Kopie
    GpuBuffer<real> old = std::move(valuesBuffer_);
    GpuBuffer<real> fresh(valuesLimit_, stream_);
    checkCudaError(cudaMemcpyAsync(fresh.get(), old.get(), bytes,
                                   cudaMemcpyDeviceToDevice, stream_));

    // 3) Copy-done-Event aufnehmen (auf stream_)
    cudaEvent_t copy_done = nullptr;
    checkCudaError(cudaEventCreateWithFlags(&copy_done, cudaEventDisableTiming));
    checkCudaError(cudaEventRecord(copy_done, stream_));

    // 4) Dem alten GC die „retiring“-Members hinlegen:
    //    - valuesBuffer_ zeigt wieder auf OLD
    //    - lastUseEvent_ zeigt auf copy_done
    valuesBuffer_ = std::move(old);
    if (lastUseEvent_) { cudaEventDestroy(lastUseEvent_); } // aufräumen, wird ersetzt
    lastUseEvent_ = copy_done;

    //    -> GC liest Members, wartet auf lastUseEvent_ (copy_done), gibt Buffer frei
    scheduleBufGC_();

    // 5) neuen aktiven Buffer einsetzen
    valuesBuffer_ = std::move(fresh);
    lastUseEvent_ = nullptr; // gleich neu recorden (nach dem Patch)
  }

  // 6) Einzelwert patchen
  int idx = getLutIndex(i, j);
  checkCudaError(cudaMemcpyAsync(valuesBuffer_.get() + idx, &value, sizeof(real),
                                 cudaMemcpyHostToDevice, stream_));

  // 7) neues lastUseEvent_ für den **aktuellen aktiven** Buffer recorden
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, stream_));
}

real InterParameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument(
      "Cannot get uniform value of non-uniform InterParameter " + name_ + ".");
  }
  return uniformValue_;
}

real InterParameter::getBetween(unsigned int i, unsigned int j) const {
  system_->checkIdxInRegions(i);
  system_->checkIdxInRegions(j);
  if (i == j) {
    throw std::invalid_argument("Can not get " + name_
            + ", region indexes must be different: " + std::to_string(i) + ".");
  }

  if (isUniform())
    return uniformValue_;
  
  // legally copy single value from device to host
  real hostValue;
  checkCudaError(cudaMemcpy(&hostValue, &valuesBuffer_.get()[getLutIndex(i, j)],
                 sizeof(real), cudaMemcpyDeviceToHost));
  return hostValue;
}

CuInterParameter InterParameter::cu() const {
  return CuInterParameter(this);
}