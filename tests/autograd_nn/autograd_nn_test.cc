// Autograd and the C++ frontend: a 2-layer MLP trained with Adam.

#include <torch/torch.h>

#include <cstdio>

#include "tests/common/check.h"

int main() {
  torch::manual_seed(0);
  torch::nn::Sequential model(
      torch::nn::Linear(16, 32),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(32, 4));
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-2));

  const torch::Tensor input = torch::randn({64, 16});
  const torch::Tensor target = torch::randn({64, 4});

  float first_loss = 0.0f;
  float last_loss = 0.0f;
  for (int step = 0; step < 100; ++step) {
    optimizer.zero_grad();
    const torch::Tensor loss = torch::mse_loss(model->forward(input), target);
    loss.backward();
    if (step == 0) {
      first_loss = loss.item<float>();
      for (const auto& parameter : model->parameters()) {
        CHECK(parameter.grad().defined());
        CHECK(parameter.grad().abs().sum().item<float>() > 0.0f);
      }
    }
    optimizer.step();
    last_loss = loss.item<float>();
  }
  std::printf("loss: %f -> %f\n", first_loss, last_loss);
  CHECK(last_loss < 0.5f * first_loss);
  return test::Finish("autograd_nn_test");
}
