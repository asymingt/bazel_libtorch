// torch_xpu registration, and its SYCL runtime (@sycl-stub's, which has no
// devices) not crashing without an Intel GPU.

#include <ATen/Context.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <torch/torch.h>
#include <torch/xpu.h>

#include <cstdio>

#include "tests/common/check.h"

int main() {
  // The XPU hooks are linked and registered.
  EXPECT(at::hasXPU());
  EXPECT(at::detail::getXPUHooks().isBuilt());

  const bool available = torch::xpu::is_available();
  const auto count = torch::xpu::device_count();
  std::printf("XPU available: %s, %d device(s)\n", available ? "yes" : "no",
              static_cast<int>(count));
  EXPECT(!available);
  EXPECT(count == 0);
  return test::Finish("xpu_dispatch_test");
}
