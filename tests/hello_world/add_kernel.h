#pragma once

#include <cstdint>
#include <string>

namespace hello_world {

// Result of running the CUDA kernel on the first GPU.
struct CudaResult {
  bool ran = false;     // False if no usable GPU (or driver) was found.
  std::string message;  // Why the kernel did not run, or the device name.
  bool correct = false; // Whether the kernel output matched the expectation.
};

// Adds two vectors of `n` floats on the GPU and checks the result.
CudaResult RunAddKernel(int64_t n);

}  // namespace hello_world
