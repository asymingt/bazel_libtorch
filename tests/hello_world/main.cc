// Links @libtorch//:c10 and, where @libtorch//:cuda_enabled holds, a CUDA
// kernel compiled by @rules_cuda. The kernel only runs if a GPU and driver are
// present at runtime, so the test passes on CPU-only machines too.

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Logging.h>

#include <cstdio>

#ifdef HELLO_WORLD_HAS_CUDA
#include "tests/hello_world/add_kernel.h"
#endif

int main() {
  const c10::Device device(c10::DeviceType::CUDA, 0);
  std::printf("Hello from c10: %s, %s\n", device.str().c_str(),
              c10::toString(c10::ScalarType::Float));

#ifdef HELLO_WORLD_HAS_CUDA
  const hello_world::CudaResult result = hello_world::RunAddKernel(1 << 20);
  if (!result.ran) {
    std::printf("CUDA kernel skipped: %s\n", result.message.c_str());
    return 0;
  }
  std::printf("CUDA kernel ran on %s: %s\n", result.message.c_str(),
              result.correct ? "correct" : "WRONG");
  return result.correct ? 0 : 1;
#else
  std::printf("Built without CUDA for this target platform\n");
  return 0;
#endif
}
