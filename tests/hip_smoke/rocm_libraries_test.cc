// The ROCm math libraries of @rocm-libraries-stub: their headers compile
// (including the device code of rocPRIM/hipCUB, rocThrust and hipRAND), and
// their shims link and, without ROCm, report errors instead of crashing.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipcub/hipcub.hpp>
#include <hipfft/hipfft.h>
#include <hiprand/hiprand_kernel.h>
#include <hipsolver/hipsolver.h>
#include <hipsparse/hipsparse.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cstdio>

#include "tests/common/check.h"

namespace {

__global__ void Uniform(unsigned long long seed, float* out) {
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, threadIdx.x, 0, &state);
  out[threadIdx.x] = hiprand_uniform(&state);
}

__global__ void BlockSum(const int* in, int* out) {
  using BlockReduce = hipcub::BlockReduce<int, 64>;
  __shared__ typename BlockReduce::TempStorage storage;
  const int sum = BlockReduce(storage).Sum(in[threadIdx.x]);
  if (threadIdx.x == 0) *out = sum;
}

}  // namespace

int main() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    // The shims return each library's "not supported" status.
    rocblas_handle rocblas = nullptr;
    EXPECT(rocblas_create_handle(&rocblas) != rocblas_status_success);
    hipblasHandle_t hipblas = nullptr;
    EXPECT(hipblasCreate(&hipblas) != HIPBLAS_STATUS_SUCCESS);
    hipblasLtHandle_t hipblaslt = nullptr;
    EXPECT(hipblasLtCreate(&hipblaslt) != HIPBLAS_STATUS_SUCCESS);
    hipsolverHandle_t hipsolver = nullptr;
    EXPECT(hipsolverCreate(&hipsolver) != HIPSOLVER_STATUS_SUCCESS);
    hipsparseHandle_t hipsparse = nullptr;
    EXPECT(hipsparseCreate(&hipsparse) != HIPSPARSE_STATUS_SUCCESS);
    hipfftHandle plan = 0;
    EXPECT(hipfftCreate(&plan) != HIPFFT_SUCCESS);
    EXPECT(rocsolver_dgetrf(rocblas, 0, 0, nullptr, 0, nullptr, nullptr) != rocblas_status_success);
    std::printf("No ROCm device: every library reports an error\n");
    return test::Finish("rocm_libraries_test");
  }

  // With an AMD GPU: device code of rocThrust (rocPRIM), hipCUB and hipRAND.
  thrust::device_vector<int> values(64, 2);
  EXPECT(thrust::reduce(values.begin(), values.end()) == 128);
  int* sum = nullptr;
  EXPECT(hipMalloc(&sum, sizeof(int)) == hipSuccess);
  BlockSum<<<1, 64>>>(thrust::raw_pointer_cast(values.data()), sum);
  int host_sum = 0;
  EXPECT(hipMemcpy(&host_sum, sum, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess);
  EXPECT(host_sum == 128);
  float* uniform = nullptr;
  EXPECT(hipMalloc(&uniform, 64 * sizeof(float)) == hipSuccess);
  Uniform<<<1, 64>>>(1234, uniform);
  float host_uniform[64];
  EXPECT(hipMemcpy(host_uniform, uniform, sizeof(host_uniform), hipMemcpyDeviceToHost) == hipSuccess);
  for (float u : host_uniform) EXPECT(u > 0.0f && u <= 1.0f);
  EXPECT(hipFree(sum) == hipSuccess);
  EXPECT(hipFree(uniform) == hipSuccess);
  return test::Finish("rocm_libraries_test");
}
