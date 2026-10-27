// c10d with Gloo (USE_DISTRIBUTED, USE_C10D_GLOO): two ranks, in two threads
// of this process, rendezvous through a FileStore and allreduce a tensor
// over Gloo's loopback transport.

#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/torch.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "tests/common/check.h"

namespace {

constexpr int kWorldSize = 2;

// Rank r contributes r + 1 everywhere, so the sum is 1 + 2 = 3.
torch::Tensor AllreduceOnRank(const std::string& store_path, int rank) {
  const auto store = c10::make_intrusive<c10d::FileStore>(store_path, kWorldSize);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
  c10d::ProcessGroupGloo group(store, rank, kWorldSize, options);
  std::vector<torch::Tensor> tensors = {torch::full({4, 4}, rank + 1.0)};
  group.allreduce(tensors)->wait();
  return tensors[0];
}

}  // namespace

int main() {
  const char* tmpdir = std::getenv("TEST_TMPDIR");
  const std::string store_path = std::string(tmpdir ? tmpdir : "/tmp") + "/gloo_store";
  std::vector<torch::Tensor> results(kWorldSize);
  std::vector<std::thread> ranks;
  for (int rank = 0; rank < kWorldSize; ++rank) {
    ranks.emplace_back([&, rank] { results[rank] = AllreduceOnRank(store_path, rank); });
  }
  for (auto& rank : ranks) rank.join();

  for (int rank = 0; rank < kWorldSize; ++rank) {
    std::printf("rank %d: allreduce sum %g\n", rank, results[rank][0][0].item<double>());
    EXPECT(torch::equal(results[rank], torch::full({4, 4}, 3.0)));
  }
  return test::Finish("distributed_gloo_test");
}
