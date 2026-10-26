// CPU tensor math: factories, matmul, FFT (pocketfft) and transcendental
// functions, checked against scalar reference values.

#include <torch/torch.h>

#include <cmath>
#include <complex>
#include <cstdio>

#include "tests/common/check.h"

namespace {

void TestFactories() {
  const torch::Tensor a = torch::arange(6, torch::kFloat32).reshape({2, 3});
  CHECK(a.sizes() == torch::IntArrayRef({2, 3}));
  CHECK(a[1][2].item<float>() == 5.0f);
  CHECK(torch::eye(4).sum().item<float>() == 4.0f);
  torch::manual_seed(0);
  const torch::Tensor r = torch::randn({1000});
  CHECK(std::abs(r.mean().item<float>()) < 0.2f);
  CHECK(std::abs(r.std().item<float>() - 1.0f) < 0.2f);
}

void TestMatmul() {
  const torch::Tensor a = torch::arange(6, torch::kFloat32).reshape({2, 3});
  const torch::Tensor b = torch::arange(12, torch::kFloat32).reshape({3, 4});
  const torch::Tensor c = torch::matmul(a, b);
  CHECK(c.sizes() == torch::IntArrayRef({2, 4}));
  // Row 1 of a is (3, 4, 5); column 3 of b is (3, 7, 11).
  CHECK(c[1][3].item<float>() == 3 * 3 + 4 * 7 + 5 * 11);
  // A larger product against a naive reference.
  torch::manual_seed(1);
  const torch::Tensor x = torch::randn({64, 48}, torch::kFloat64);
  const torch::Tensor y = torch::randn({48, 32}, torch::kFloat64);
  const torch::Tensor reference = (x.unsqueeze(2) * y.unsqueeze(0)).sum(1);
  CHECK(torch::allclose(torch::matmul(x, y), reference));
  CHECK(torch::allclose(torch::eye(5, torch::kFloat64).matmul(x.narrow(0, 0, 5)), x.narrow(0, 0, 5)));
}

void TestFFT() {
  constexpr int64_t n = 16;
  const torch::Tensor t = torch::arange(n, torch::kFloat64);
  const torch::Tensor signal = torch::cos(2 * M_PI * 3 * t / n);
  const torch::Tensor spectrum = torch::fft::fft(signal);
  // A cosine at bin 3 has energy n/2 at bins 3 and n-3 only.
  const torch::Tensor magnitude = spectrum.abs();
  CHECK(std::abs(magnitude[3].item<double>() - n / 2.0) < 1e-9);
  CHECK(std::abs(magnitude[n - 3].item<double>() - n / 2.0) < 1e-9);
  CHECK(std::abs(magnitude.sum().item<double>() - n) < 1e-9);
  CHECK(torch::allclose(torch::real(torch::fft::ifft(spectrum)), signal));
}

void TestTranscendentals() {
  // Long enough to exercise the vectorized loops and their scalar tails.
  const torch::Tensor x = torch::linspace(-3, 3, 1031, torch::kFloat32);
  const torch::Tensor sin = torch::sin(x);
  const torch::Tensor exp = torch::exp(x);
  const torch::Tensor erf = torch::erf(x);
  float max_error = 0.0f;
  for (int64_t i = 0; i < x.numel(); ++i) {
    const double v = x[i].item<float>();
    max_error = std::max<float>(max_error, std::abs(sin[i].item<float>() - std::sin(v)));
    max_error = std::max<float>(max_error, std::abs(exp[i].item<float>() - std::exp(v)) / std::exp(v));
    max_error = std::max<float>(max_error, std::abs(erf[i].item<float>() - std::erf(v)));
  }
  std::printf("max error of sin/exp/erf: %g\n", max_error);
  CHECK(max_error < 1e-5f);
}

}  // namespace

int main() {
  TestFactories();
  TestMatmul();
  TestFFT();
  TestTranscendentals();
  return test::Finish("tensor_math_test");
}
