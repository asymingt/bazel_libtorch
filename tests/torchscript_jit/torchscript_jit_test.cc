// TorchScript: compile a module from source, run it, and round-trip it
// through serialization.

#include <torch/script.h>
#include <torch/torch.h>

#include <sstream>

#include "tests/common/check.h"

int main() {
  torch::jit::Module module("Model");
  module.define(R"(
def forward(self, x, y):
    return torch.relu(x @ y + 1.0)
)");

  torch::manual_seed(0);
  const torch::Tensor x = torch::randn({4, 8});
  const torch::Tensor y = torch::randn({8, 3});
  const torch::Tensor expected = torch::relu(torch::matmul(x, y) + 1.0);

  const torch::Tensor output = module.forward({x, y}).toTensor();
  CHECK(torch::allclose(output, expected));

  std::stringstream buffer;
  module.save(buffer);
  torch::jit::Module loaded = torch::jit::load(buffer);
  const torch::Tensor reloaded_output = loaded.forward({x, y}).toTensor();
  CHECK(torch::equal(reloaded_output, output));

  // A free function compiled from source.
  auto unit = torch::jit::compile(R"(
def scale(x, factor: float):
    return x * factor
)");
  CHECK(torch::allclose(unit->run_method("scale", x, 2.0).toTensor(), x * 2.0));
  return test::Finish("torchscript_jit_test");
}
