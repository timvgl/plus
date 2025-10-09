#include "field.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "system.hpp"

namespace {
  struct FieldCleanup {
    GpuBuffer<real*> bufferPtrs;
    std::vector<GpuBuffer<real>> buffers;
    cudaEvent_t lastUseEvent = nullptr;
  };

  static void CUDART_CB field_cleanup_cb(void* p) noexcept {
    auto* cap = static_cast<FieldCleanup*>(p);

    if (cap->lastUseEvent) {
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap;
  }
}

Field::~Field() {
  // Nichts zu tun?
  if (!has_buffers_()) {
    if (lastUseEvent_) {
      cudaEventDestroy(lastUseEvent_);
      lastUseEvent_ = nullptr;
    }
    return;
  }

  auto* cap = new (std::nothrow) FieldCleanup{};
  if (!cap) {
    if (lastUseEvent_) {
      cudaEventSynchronize(lastUseEvent_);
      cudaEventDestroy(lastUseEvent_);
      lastUseEvent_ = nullptr;
    }
    free();
    printf("Warning: Field destructor failed to allocate cleanup structure, falling back to synchronous cleanup.\n");
    return;
  }

  cap->buffers       = std::move(buffers_);
  cap->bufferPtrs    = std::move(bufferPtrs_);
  cap->lastUseEvent  = lastUseEvent_;
  lastUseEvent_      = nullptr;

  cudaStream_t s_reaper = getCudaStreamGC();

  if (cap->lastUseEvent) {
    checkCudaError(cudaStreamWaitEvent(s_reaper, cap->lastUseEvent, 0));
  }

  cudaError_t st = cudaLaunchHostFunc(s_reaper, field_cleanup_cb, cap);
  if (st != cudaSuccess) {
    if (cap->lastUseEvent) {
      cudaEventSynchronize(cap->lastUseEvent);
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap;
  }
}

void Field::ensureReadyOn(cudaStream_t consumer) const {
  if (lastUseEvent_) {
    checkCudaError(cudaStreamWaitEvent(consumer, lastUseEvent_, 0));
  }
}


void Field::markLastUse() const {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, stream_));
}

void Field::markLastUse(cudaStream_t s) const {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}

Field::Field() : system_(nullptr), ncomp_(0), stream_(getCudaStream()) {}


Field::Field(std::shared_ptr<const System> system, int nComponents)
    : system_(system), ncomp_(nComponents), stream_(getCudaStream()) {
  allocate();
  setZeroOutsideGeometry();
}

Field::Field(std::shared_ptr<const System> system, int nComponents, cudaStream_t s)
    : system_(system), ncomp_(nComponents), stream_(s) {
  allocate();
  setZeroOutsideGeometry();
}

Field::Field(std::shared_ptr<const System> system, int nComponents, real value)
    : Field(system, nComponents, getCudaStream()) {
  setUniformValue(value);
}

Field::Field(std::shared_ptr<const System> system, int nComponents, real value, cudaStream_t s)
    : Field(system, nComponents, s) {
  setUniformValue(value);
}

Field::Field(std::shared_ptr<const System> system, int nComponents, real3 value)
    : Field(system, nComponents, getCudaStream()) {
  setUniformValue(value);
}

Field::Field(std::shared_ptr<const System> system, int nComponents, real3 value, cudaStream_t s)
    : Field(system, nComponents, s) {
  setUniformValue(value);
}

Field Field::eval() const { return Field(*this); } // echtes Override

//Field::Field(const Field& other) : system_(other.system_), ncomp_(other.ncomp_) {
//  buffers_ = other.buffers_;
//  updateDevicePointersBuffer();
//}

/* Field::Field(const Field& other)
  : system_(other.system_), ncomp_(other.ncomp_), stream_(other.stream_) {
  buffers_.resize(other.buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    buffers_[i] = GpuBuffer<real>(other.buffers_[i].size(), stream_);
    checkCudaError(cudaMemcpyAsync(
      buffers_[i].get(), other.buffers_[i].get(),
      buffers_[i].size() * sizeof(real),
      cudaMemcpyDeviceToDevice, stream_    // <— nicht getCudaStream()
    ));
  }
  updateDevicePointersBuffer();
} */

Field::Field(const Field& other)
  : system_(other.system_), ncomp_(other.ncomp_), stream_(other.stream_) {
  buffers_.resize(other.buffers_.size());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    buffers_[i] = GpuBuffer<real>(other.buffers_[i].size(), stream_);
    checkCudaError(cudaMemcpyAsync(
      buffers_[i].get(), other.buffers_[i].get(),
      buffers_[i].size() * sizeof(real),
      cudaMemcpyDeviceToDevice, stream_
    ));
  }
  updateDevicePointersBuffer();
  lastUseEvent_ = nullptr;
}


//Field::Field(Field&& other) : system_(other.system_), ncomp_(other.ncomp_) {
//  buffers_ = std::move(other.buffers_);
//  bufferPtrs_ = std::move(other.bufferPtrs_);
//  other.clear();
//}

/* Field::Field(Field&& other)
  : ncomp_(other.ncomp_),
    system_(std::move(other.system_)),
    buffers_(std::move(other.buffers_)),
    bufferPtrs_(std::move(other.bufferPtrs_)),
    stream_(other.stream_)
{
  other.ncomp_ = 0;
  other.system_.reset();
  other.buffers_.clear();
  other.bufferPtrs_ = GpuBuffer<real*>{};  // default
  other.stream_ = nullptr;
} */

Field::Field(Field&& other)
  : ncomp_(other.ncomp_),
    system_(std::move(other.system_)),
    buffers_(std::move(other.buffers_)),
    bufferPtrs_(std::move(other.bufferPtrs_)),
    stream_(other.stream_),
    lastUseEvent_(other.lastUseEvent_)
{
  other.ncomp_ = 0;
  other.system_.reset();
  other.buffers_.clear();
  other.bufferPtrs_ = GpuBuffer<real*>{};
  other.stream_ = nullptr;
  other.lastUseEvent_ = nullptr;
}

//Field& Field::operator=(const Field& other) {
//  if (this == &other)
//    return *this;
//  return *this = std::move(Field(other));  // moves a copy of other to this
//}

Field& Field::operator=(const Field& other) {
  if (this == &other) return *this;
  Field tmp(other);           // deep copy
  // move-assign (siehe korrigiertes Move unten)
  return *this = std::move(tmp);
}

Field& Field::operator=(const FieldQuantity& q) {
  Field tmp = q.eval();      // tmp hat irgendeinen stream_
  tmp.stream_ = this->stream_;   // angleichen
  return *this = std::move(tmp); // move-assign
}

//Field& Field::operator=(Field&& other) {
//  system_ = other.system_;
//  ncomp_ = other.ncomp_;
//  buffers_ = std::move(other.buffers_);
//  bufferPtrs_ = std::move(other.bufferPtrs_);
//  other.clear();
//  return *this;
//}

/* Field& Field::operator=(Field&& other) {
  if (this != &other) {
    stream_      = other.stream_;
    ncomp_       = other.ncomp_;
    system_      = std::move(other.system_);
    buffers_     = std::move(other.buffers_);
    bufferPtrs_  = std::move(other.bufferPtrs_);

    other.ncomp_ = 0;
    other.system_.reset();
    other.buffers_.clear();
    other.bufferPtrs_ = GpuBuffer<real*>{};
    other.stream_ = nullptr;
  }
  return *this;
} */

Field& Field::operator=(Field&& other) {
  if (this != &other) {
    // Eigenes laufendes Event sicher beenden
    if (lastUseEvent_) {
      cudaEventSynchronize(lastUseEvent_);
      cudaEventDestroy(lastUseEvent_);
      lastUseEvent_ = nullptr;
    }
    // Eigene Ressourcen freigeben
    free();

    // Übernahme
    stream_      = other.stream_;
    ncomp_       = other.ncomp_;
    system_      = std::move(other.system_);
    buffers_     = std::move(other.buffers_);
    bufferPtrs_  = std::move(other.bufferPtrs_);
    lastUseEvent_ = other.lastUseEvent_;   // <--- NEU

    // Donor invalidieren
    other.ncomp_ = 0;
    other.system_.reset();
    other.buffers_.clear();
    other.bufferPtrs_ = GpuBuffer<real*>{};
    other.stream_ = nullptr;
    other.lastUseEvent_ = nullptr;         // <--- NEU
  }
  return *this;
}

/* void Field::clear() {
  system_ = nullptr;
  ncomp_ = 0;
  free();
} */

void Field::clear() {
  // Falls noch ein Event aussteht, darauf warten & zerstören
  if (lastUseEvent_) {
    cudaEventSynchronize(lastUseEvent_);
    cudaEventDestroy(lastUseEvent_);
    lastUseEvent_ = nullptr;
  }
  system_ = nullptr;
  ncomp_ = 0;
  free();
}

std::shared_ptr<const System> Field::system() const {
  return system_;
}

void Field::updateDevicePointersBuffer() {
  std::vector<real*> bufferPtrsOnHost(ncomp_);
  std::transform(buffers_.begin(), buffers_.end(), bufferPtrsOnHost.begin(),
                 [](auto& buf) { return buf.get(); });
  bufferPtrs_ = GpuBuffer<real*>(bufferPtrsOnHost, stream_);
}

void Field::allocate() {
  free();

  if(empty())
    return;

  buffers_ =
      std::vector<GpuBuffer<real>>(ncomp_, GpuBuffer<real>(grid().ncells(), stream_));

  updateDevicePointersBuffer();
}

void Field::free() {
  buffers_.clear();
  bufferPtrs_.recycle();
}

CuField Field::cu() const {
  return CuField(this);
}

void Field::getData(real* buffer) const {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid().ncells();
    checkCudaError(cudaMemcpyAsync(bufferComponent, buffers_[c].get(),
                                   grid().ncells() * sizeof(real),
                                   cudaMemcpyDeviceToHost, stream_));
  }
}

std::vector<real> Field::getData() const {
  auto size = ncomp_ * grid().ncells();
  std::vector<real> buffer(size, 0);
  getData(buffer.data());

  return buffer;
}

void Field::setData(const real* buffer) {
  for (int c = 0; c < ncomp_; c++) {
    auto bufferComponent = buffer + c * grid().ncells();
    checkCudaError(cudaMemcpyAsync(buffers_[c].get(), bufferComponent,
                                   grid().ncells() * sizeof(real),
                                   cudaMemcpyHostToDevice, stream_));
  }
  setZeroOutsideGeometry();
}

void Field::setData(const real* buffer, cudaStream_t s) {
  for (int c = 0; c < ncomp_; c++) {
    auto bufferComponent = buffer + c * grid().ncells();
    checkCudaError(cudaMemcpyAsync(buffers_[c].get(), bufferComponent,
                                   grid().ncells() * sizeof(real),
                                   cudaMemcpyHostToDevice, s));
  }
  setZeroOutsideGeometry();
}

void Field::setData(const std::vector<real>& buffer) {
  setData(buffer.data());
}

void Field::setData(const std::vector<real>& buffer, cudaStream_t s) {
  setData(buffer.data(), s);
}

__global__ void k_setComponent(CuField f, real value, int comp) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;

  if (f.cellInGeometry(idx)) {
    f.setValueInCell(idx, comp, value);
  } else {
    f.setValueInCell(idx, comp, 0.0);
  }
}

__global__ void k_setComponentInRegion(CuField f,
                                       real value,
                                       int comp,
                                       unsigned int region_idx) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInRegion(region_idx, idx))
    return;
  if (f.cellInGeometry(idx))
    f.setValueInCell(idx, comp, value);
  else
    f.setValueInCell(idx, comp, 0.0);
}

__global__ void k_setVectorValue(CuField f, real3 value) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;

  if (f.cellInGeometry(idx)) {
    f.setVectorInCell(idx, value);
  } else {
    f.setVectorInCell(idx, real3{0, 0, 0});
  }
}

__global__ void k_setVectorValueInRegion(CuField f, real3 value, unsigned int region_idx) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (!f.cellInGrid(idx) || !f.cellInRegion(region_idx, idx))
    return;

  if (f.cellInGeometry(idx)) {
    f.setVectorInCell(idx, value);
  } else {
    f.setVectorInCell(idx, real3{0, 0, 0});
  }
}

void Field::setUniformComponent(int comp, real value) {
  cudaLaunchOn(stream_, "field.cu", grid().ncells(), k_setComponent, cu(), value, comp);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformComponent(int comp, real value, cudaStream_t s) {
  cudaLaunchOn(s, "field.cu", grid().ncells(), k_setComponent, cu(), value, comp);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformComponentInRegion(unsigned int regionIdx, int comp, real value) {
  system_->checkIdxInRegions(regionIdx);
  cudaLaunchOn(stream_, "field.cu", grid().ncells(), k_setComponentInRegion, cu(), value, comp, regionIdx);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformComponentInRegion(unsigned int regionIdx, int comp, real value, cudaStream_t s) {
  system_->checkIdxInRegions(regionIdx);
  cudaLaunchOn(s, "field.cu", grid().ncells(), k_setComponentInRegion, cu(), value, comp, regionIdx);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformValue(real value) {
  for (int comp = 0; comp < ncomp_; comp++)
    setUniformComponent(comp, value);
}

void Field::setUniformValue(real value, cudaStream_t s) {
  for (int comp = 0; comp < ncomp_; comp++)
    setUniformComponent(comp, value, s);
}


void Field::setUniformValue(real3 value) {
  cudaLaunchOn(stream_, "field.cu", grid().ncells(), k_setVectorValue, cu(), value);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformValue(real3 value, cudaStream_t s) {
  cudaLaunchOn(s, "field.cu", grid().ncells(), k_setVectorValue, cu(), value);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformValueInRegion(unsigned int regionIdx, real value) {
  system_->checkIdxInRegions(regionIdx);
  for (int comp = 0; comp < ncomp_; comp++)
    setUniformComponentInRegion(regionIdx, comp, value);
}

void Field::setUniformValueInRegion(unsigned int regionIdx, real value, cudaStream_t s) {
  system_->checkIdxInRegions(regionIdx);
  for (int comp = 0; comp < ncomp_; comp++)
    setUniformComponentInRegion(regionIdx, comp, value, s);
}

void Field::setUniformValueInRegion(unsigned int regionIdx, real3 value) {
  system_->checkIdxInRegions(regionIdx);
  cudaLaunchOn(stream_, "field.cu", grid().ncells(), k_setVectorValueInRegion, cu(), value, regionIdx);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::setUniformValueInRegion(unsigned int regionIdx, real3 value, cudaStream_t s) {
  system_->checkIdxInRegions(regionIdx);
  cudaLaunchOn(s, "field.cu", grid().ncells(), k_setVectorValueInRegion, cu(), value, regionIdx);
  //checkCudaError(cudaDeviceSynchronize());
}

void Field::makeZero() {
  setUniformValue(0);
}

void Field::makeZero(cudaStream_t s) {
  setUniformValue(0, s);
}

__global__ void k_setZeroOutsideGeometry(CuField f) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (f.cellInGrid(idx) && !f.cellInGeometry(idx)) {
    for (int comp = 0; comp < f.ncomp; comp++)
      f.setValueInCell(idx, comp, 0.0);
  }
}

void Field::setZeroOutsideGeometry() {
  if (!system_)
    return;
  if (system_->geometry().size() > 0)
    cudaLaunchOn(stream_, "field.cu", grid().ncells(), k_setZeroOutsideGeometry, cu());
    //checkCudaError(cudaDeviceSynchronize());
}

Field& Field::operator+=(const Field& other) {
  addTo(*this, 1, other);
  return *this;
}

Field& Field::operator-=(const Field& other) {
  addTo(*this, -1, other);
  return *this;
}

Field& Field::operator+=(const FieldQuantity& q) {
  if (!q.assuredZero())
    addTo(*this, 1, q.eval());
  return *this;
}

Field& Field::operator-=(const FieldQuantity& q) {
  if (!q.assuredZero())
    addTo(*this, -1, q.eval());
  return *this;
}
