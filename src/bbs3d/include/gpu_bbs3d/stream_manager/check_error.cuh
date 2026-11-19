#pragma once

#include <string>
#include <iostream>

#include <cuda_runtime.h>

namespace lightning::gpu {

class CUDACheckError {
public:
  void operator<<(cudaError_t error) const;
};

extern CUDACheckError check_error;

}  // namespace gpu