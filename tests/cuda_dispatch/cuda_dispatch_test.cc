// torch_cuda registration, and (with an NVIDIA GPU) the CUDA kernels and
// libraries checked against the CPU: cuBLAS, cuSOLVER, cuFFT and the
// CUTLASS-based memory-efficient attention.

#include <ATen/Context.h>
#include <torch/torch.h>

#include <cstdio>

#include "tests/common/check.h"

namespace {

void TestOnGpu() {
  const torch::Device cuda(torch::kCUDA, 0);
  torch::manual_seed(0);

  const torch::Tensor a = torch::randn({128, 64});
  const torch::Tensor b = torch::randn({64, 32});
  CHECK(torch::allclose(torch::matmul(a.to(cuda), b.to(cuda)).cpu(),
                        torch::matmul(a, b), 1e-3, 1e-3));

  const torch::Tensor m = torch::randn({16, 8}, torch::kFloat64);
  const torch::Tensor s_cuda = std::get<1>(torch::linalg_svd(m.to(cuda), false));
  const torch::Tensor s_cpu = torch::linalg_svdvals(m);
  CHECK(torch::allclose(s_cuda.cpu(), s_cpu));

  const torch::Tensor signal = torch::randn({256}, torch::kFloat64);
  CHECK(torch::allclose(torch::fft::fft(signal.to(cuda)).cpu(),
                        torch::fft::fft(signal)));

  // (batch, heads, sequence, head dimension), on the memory-efficient kernel.
  const torch::Tensor q = torch::randn({2, 4, 64, 32});
  const torch::Tensor k = torch::randn({2, 4, 64, 32});
  const torch::Tensor v = torch::randn({2, 4, 64, 32});
  const torch::Tensor expected = torch::scaled_dot_product_attention(q, k, v);
  const auto [output, log_sum_exp, philox_seed, philox_offset] =
      at::_scaled_dot_product_efficient_attention(
          q.to(cuda), k.to(cuda), v.to(cuda), std::nullopt,
          /*compute_log_sumexp=*/false);
  CHECK(torch::allclose(output.cpu(), expected, 1e-3, 1e-3));
}

}  // namespace

int main() {
  // CUDAHooks and the CUDA kernels are linked and registered.
  CHECK(at::hasCUDA() || !torch::cuda::is_available());
  CHECK(at::detail::getCUDAHooks().hasCUDART());

  // Must not crash without a GPU or driver.
  const bool available = torch::cuda::is_available();
  const auto count = torch::cuda::device_count();
  std::printf("CUDA available: %s, %d device(s)\n", available ? "yes" : "no",
              static_cast<int>(count));
  if (available) {
    TestOnGpu();
  }
  return test::Finish("cuda_dispatch_test");
}
