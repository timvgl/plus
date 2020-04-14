#include <map>
#include <vector>

#include "datatypes.hpp"

class BufferPool {
 public:
  BufferPool() = default;
  BufferPool(BufferPool const&) = delete;
  void operator=(BufferPool const&) = delete;

  real* allocate(int size);
  void free(real*&);
  void recycle(real*&);

 private:
  std::map<int, std::vector<real*>> pool_;
  std::map<real*, int> inUse_;
};

extern BufferPool bufferPool;