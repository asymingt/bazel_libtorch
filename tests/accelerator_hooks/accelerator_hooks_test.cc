// The Meta device, PrivateUse1 registration, and the accelerator hook
// interfaces that GPU backends and out-of-tree plugins register at runtime.

#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/detail/HIPHooksInterface.h>
#include <ATen/detail/HPUHooksInterface.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/detail/MTIAHooksInterface.h>
#include <ATen/detail/XLAHooksInterface.h>
#include <ATen/detail/XPUHooksInterface.h>
#include <torch/torch.h>

#include "tests/common/check.h"

int main() {
  // Meta tensors propagate shapes and dtypes without allocating storage.
  const torch::Tensor a = torch::empty({1024, 1024}, torch::kMeta);
  const torch::Tensor b = torch::empty({1024, 256}, torch::kMeta);
  const torch::Tensor c = torch::matmul(a, b);
  CHECK(c.is_meta());
  CHECK(c.sizes() == torch::IntArrayRef({1024, 256}));
  CHECK(c.dtype() == torch::kFloat32);
  CHECK(c.storage().data() == nullptr);

  c10::register_privateuse1_backend("my_npu");
  CHECK(c10::get_privateuse1_backend() == "my_npu");
  CHECK(c10::Device("my_npu:1").is_privateuseone());

  // Without a backend registered, every hook interface is the default stub,
  // which reports the backend as unavailable instead of failing.
  CHECK(!at::detail::getCUDAHooks().hasCUDA());
  CHECK(!at::detail::getHIPHooks().hasHIP());
  CHECK(!at::detail::getXPUHooks().hasXPU());
  CHECK(!at::detail::getMPSHooks().hasMPS());
  CHECK(!at::detail::getHPUHooks().isAvailable());
  CHECK(!at::detail::getMTIAHooks().isAvailable());
  CHECK(!at::detail::getXLAHooks().hasXLA());
  CHECK(!torch::cuda::is_available());
  return test::Finish("accelerator_hooks_test");
}
