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
  EXPECT(c.is_meta());
  EXPECT(c.sizes() == torch::IntArrayRef({1024, 256}));
  EXPECT(c.dtype() == torch::kFloat32);
  EXPECT(c.storage().data() == nullptr);

  c10::register_privateuse1_backend("my_npu");
  EXPECT(c10::get_privateuse1_backend() == "my_npu");
  EXPECT(c10::Device("my_npu:1").is_privateuseone());

  // Without a backend registered, every hook interface is the default stub,
  // which reports the backend as unavailable instead of failing.
  EXPECT(!at::detail::getCUDAHooks().hasCUDA());
  EXPECT(!at::detail::getHIPHooks().hasHIP());
  EXPECT(!at::detail::getXPUHooks().hasXPU());
#ifndef TEST_MPS_ENABLED
  EXPECT(!at::detail::getMPSHooks().hasMPS());
#endif
  EXPECT(!at::detail::getHPUHooks().isAvailable());
  EXPECT(!at::detail::getMTIAHooks().isAvailable());
  EXPECT(!at::detail::getXLAHooks().hasXLA());
  EXPECT(!torch::cuda::is_available());
  return test::Finish("accelerator_hooks_test");
}
