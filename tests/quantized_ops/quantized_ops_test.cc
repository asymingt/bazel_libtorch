// Quantized INT8 operators on FBGEMM (USE_FBGEMM): a quantized linear layer
// and max pooling, checked against their float counterparts.
//
// FBGEMM's INT8 GEMM (PackBMatrix) has only x86 kernels: on AArch64 it is
// built for its FP16/FP32 kernels, and upstream runs quantized linear layers
// there on QNNPACK or oneDNN, neither of which is built yet. So the linear
// layer is only checked on x86_64.

#include <ATen/Context.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>

#include "tests/common/check.h"

namespace {

// Calls a quantized:: operator through the dispatcher, as TorchScript does.
c10::IValue Call(const char* name, std::vector<c10::IValue> stack) {
  const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, "");
  op.callBoxed(&stack);
  return stack.at(0);
}

}  // namespace

int main() {
  const auto& engines = at::globalContext().supportedQEngines();
  EXPECT(std::find(engines.begin(), engines.end(), at::QEngine::FBGEMM) != engines.end());
  at::globalContext().setQEngine(at::QEngine::FBGEMM);
  torch::manual_seed(0);

#if defined(__x86_64__) || defined(_M_X64)
  // y = x W^T + b, with x quantized to quint8 and W to qint8.
  const torch::Tensor x = torch::rand({8, 32});
  const torch::Tensor w = torch::randn({16, 32}) * 0.1;
  const torch::Tensor b = torch::randn({16});
  const torch::Tensor qx = at::quantize_per_tensor(x, 1.0 / 255, 0, torch::kQUInt8);
  const torch::Tensor qw = at::quantize_per_tensor(w, 0.5 / 127, 0, torch::kQInt8);
  const c10::IValue packed = Call("quantized::linear_prepack", {qw, b});
  const torch::Tensor expected = torch::linear(x, w, b);
  const double y_scale = (expected.max() - expected.min()).item<double>() / 255;
  const int64_t y_zero_point = static_cast<int64_t>(-expected.min().item<double>() / y_scale);
  const torch::Tensor qy = Call("quantized::linear", {qx, packed, y_scale, y_zero_point}).toTensor();
  EXPECT(qy.is_quantized());
  const double linear_error = (qy.dequantize() - expected).abs().max().item<double>();
  std::printf("quantized linear: max error %g (output scale %g)\n", linear_error, y_scale);
  EXPECT(linear_error < 4 * y_scale);
#endif

  // Max pooling commutes with (monotonic) quantization.
  const torch::Tensor image = torch::rand({1, 3, 8, 8});
  const torch::Tensor qimage = at::quantize_per_tensor(image, 1.0 / 255, 0, torch::kQUInt8);
  // (2x2 windows, stride 2, no padding or dilation.)
  const torch::Tensor pooled = at::quantized_max_pool2d(qimage, {2, 2}, {2, 2}, {0, 0}, {1, 1});
  EXPECT(torch::equal(pooled.dequantize(), torch::max_pool2d(qimage.dequantize(), {2, 2}, {2, 2})));
  return test::Finish("quantized_ops_test");
}
