#include "tests/hello_world/add_kernel.h"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <cuda_runtime.h>

#include <vector>

namespace hello_world {
namespace {

__global__ void AddKernel(const float* a, const float* b, float* out,
                          int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

#define CUDA_OK(expr) TORCH_CHECK((expr) == cudaSuccess, #expr " failed")

}  // namespace

CudaResult RunAddKernel(int64_t n) {
  CudaResult result;
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    result.message = status != cudaSuccess ? cudaGetErrorString(status)
                                           : "no CUDA devices";
    return result;
  }
  cudaDeviceProp prop;
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  result.message = prop.name;

  std::vector<float> a(n), b(n), out(n);
  for (const auto i : c10::irange(n)) {
    a[i] = static_cast<float>(i);
    b[i] = 2.0f * static_cast<float>(i);
  }
  const size_t bytes = n * sizeof(float);
  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  CUDA_OK(cudaMalloc(&d_a, bytes));
  CUDA_OK(cudaMalloc(&d_b, bytes));
  CUDA_OK(cudaMalloc(&d_out, bytes));
  CUDA_OK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  AddKernel<<<blocks, kThreads>>>(d_a, d_b, d_out, n);
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  CUDA_OK(cudaFree(d_a));
  CUDA_OK(cudaFree(d_b));
  CUDA_OK(cudaFree(d_out));

  result.ran = true;
  result.correct = true;
  for (const auto i : c10::irange(n)) {
    if (out[i] != 3.0f * static_cast<float>(i)) {
      result.correct = false;
    }
  }
  return result;
}

}  // namespace hello_world
