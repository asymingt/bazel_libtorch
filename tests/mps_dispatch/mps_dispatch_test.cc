// The Apple MPS (Metal) backend: registered on Apple Silicon builds, and
// running its kernels (shaders compiled at runtime) when Metal is available.

#include <ATen/detail/MPSHooksInterface.h>
#include <torch/mps.h>
#include <torch/torch.h>

#include <cstdio>

#include "tests/common/check.h"

int main() {
  EXPECT(at::detail::getMPSHooks().isBuilt());
  const bool available = torch::mps::is_available();
  std::printf("MPS available: %s\n", available ? "yes" : "no");
  if (available) {
    const torch::Device mps(torch::kMPS);
    torch::manual_seed(0);
    const torch::Tensor a = torch::randn({64, 32});
    const torch::Tensor b = torch::randn({32, 16});
    EXPECT(torch::allclose((a.to(mps) + 1.0).cpu(), a + 1.0));
    EXPECT(torch::allclose(torch::matmul(a.to(mps), b.to(mps)).cpu(),
                           torch::matmul(a, b), 1e-3, 1e-3));
    // A kernel from an embedded Metal shader (compiled at runtime).
    EXPECT(torch::allclose(torch::erfinv(a.clamp(-0.9, 0.9).to(mps)).cpu(),
                           torch::erfinv(a.clamp(-0.9, 0.9)), 1e-3, 1e-3));
  }
  return test::Finish("mps_dispatch_test");
}
