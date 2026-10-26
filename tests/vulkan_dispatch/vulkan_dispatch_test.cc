// The Vulkan compute backend: available (and running its compute shaders) when
// a Vulkan driver is installed, and cleanly unavailable otherwise.

#include <ATen/vulkan/Context.h>
#include <torch/torch.h>

#include <cstdio>

#include "tests/common/check.h"

int main() {
  const bool available = at::native::is_vulkan_available();
  std::printf("Vulkan available: %s\n", available ? "yes" : "no");
  if (available) {
    torch::manual_seed(0);
    const torch::Tensor a = torch::rand({1, 3, 32, 32});
    const torch::Tensor b = torch::rand({1, 3, 32, 32});
    const torch::Tensor sum = (a.vulkan() + b.vulkan()).cpu();
    EXPECT(torch::allclose(sum, a + b, 1e-4, 1e-4));
    const torch::Tensor relu = torch::relu(a.vulkan() - 0.5).cpu();
    EXPECT(torch::allclose(relu, torch::relu(a - 0.5), 1e-4, 1e-4));
  }
  return test::Finish("vulkan_dispatch_test");
}
