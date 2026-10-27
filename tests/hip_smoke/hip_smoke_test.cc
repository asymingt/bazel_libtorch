// The hermetic HIP toolchain: device code compiled by Clang for AMD GPUs and
// linked against @hip-stub's libamdhip64 shim, which runs the kernel on an AMD
// GPU when ROCm is installed and reports no device otherwise.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "tests/common/check.h"

namespace {

// Exercises the device libraries (sinf and expf come from ocml).
__global__ void SinPlusExp(float* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = sinf(x[i]) + expf(x[i]);
}

}  // namespace

int main() {
  int count = -1;
  const hipError_t status = hipGetDeviceCount(&count);
  std::printf("HIP: %s, %d device(s)\n", hipGetErrorString(status), count);
  EXPECT(count >= 0);
  if (status != hipSuccess || count == 0) {
    EXPECT(count == 0);
    return test::Finish("hip_smoke_test");
  }

  constexpr int kSize = 1000;
  std::vector<float> host(kSize);
  for (int i = 0; i < kSize; ++i) host[i] = 0.001f * i;
  float* device = nullptr;
  EXPECT(hipMalloc(&device, kSize * sizeof(float)) == hipSuccess);
  EXPECT(hipMemcpy(device, host.data(), kSize * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  SinPlusExp<<<(kSize + 255) / 256, 256>>>(device, kSize);
  EXPECT(hipGetLastError() == hipSuccess);
  std::vector<float> result(kSize);
  EXPECT(hipMemcpy(result.data(), device, kSize * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  EXPECT(hipFree(device) == hipSuccess);
  for (int i = 0; i < kSize; ++i) {
    EXPECT(std::fabs(result[i] - (std::sin(host[i]) + std::exp(host[i]))) < 1e-5f);
  }
  return test::Finish("hip_smoke_test");
}
