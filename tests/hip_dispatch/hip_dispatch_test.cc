// torch_hip registration (as the CUDA backend, which ROCm masquerades as),
// and with an AMD GPU and ROCm, the hipified kernels and the rocBLAS,
// rocSOLVER and hipFFT paths checked against the CPU.

#include <ATen/Context.h>
#include <torch/torch.h>

#include <cstdio>

#include "tests/common/check.h"

namespace {

void TestOnGpu() {
  const torch::Device gpu(torch::kCUDA, 0);
  torch::manual_seed(0);

  const torch::Tensor a = torch::randn({128, 64});
  const torch::Tensor b = torch::randn({64, 32});
  EXPECT(torch::allclose((a.to(gpu) + 1.0).cpu(), a + 1.0));
  EXPECT(torch::allclose(torch::matmul(a.to(gpu), b.to(gpu)).cpu(),
                         torch::matmul(a, b), 1e-3, 1e-3));

  // rocSOLVER's SVD reconstructs its input (the CPU has no LAPACK).
  const torch::Tensor m = torch::randn({16, 8}, torch::kFloat64);
  const auto [u, s, vh] = torch::linalg_svd(m.to(gpu), false);
  EXPECT(torch::allclose(torch::matmul(u * s.unsqueeze(0), vh).cpu(), m));

  const torch::Tensor signal = torch::randn({256}, torch::kFloat64);
  EXPECT(torch::allclose(torch::fft::fft(signal.to(gpu)).cpu(),
                         torch::fft::fft(signal)));
}

}  // namespace

int main() {
  // The HIP hooks are linked and registered, and report a ROCm build.
  EXPECT(at::detail::getCUDAHooks().hasROCM());
  EXPECT(!at::detail::getCUDAHooks().hasCUDART());

  // Must not crash without a GPU or ROCm.
  const bool available = torch::cuda::is_available();
  const auto count = torch::cuda::device_count();
  std::printf("ROCm available: %s, %d device(s)\n", available ? "yes" : "no",
              static_cast<int>(count));
  EXPECT(available == (count > 0));
  if (available) {
    TestOnGpu();
  }
  return test::Finish("hip_dispatch_test");
}
