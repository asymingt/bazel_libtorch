// The Kineto profiler (USE_KINETO): a CPU trace of a tensor workload, exported
// as a Chrome trace with the workload's operators as cpu_op events.

#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/torch.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "tests/common/check.h"

namespace {

std::size_t Count(const std::string& text, const std::string& pattern) {
  std::size_t count = 0;
  for (auto i = text.find(pattern); i != std::string::npos;
       i = text.find(pattern, i + pattern.size())) {
    ++count;
  }
  return count;
}

}  // namespace

int main() {
  namespace profiler = torch::autograd::profiler;
  using torch::profiler::impl::ActivityType;
  using torch::profiler::impl::ProfilerConfig;
  using torch::profiler::impl::ProfilerState;

  const ProfilerConfig config(ProfilerState::KINETO);
  const std::set<ActivityType> activities = {ActivityType::CPU};
  profiler::prepareProfiler(config, activities);
  profiler::enableProfiler(config, activities);
  {
    const torch::Tensor a = torch::randn({64, 64});
    const torch::Tensor b = torch::randn({64, 64});
    const torch::Tensor c = torch::relu(torch::matmul(a, b) + 1.0);
    EXPECT(c.size(0) == 64);
  }
  const auto result = profiler::disableProfiler();
  EXPECT(result != nullptr);

  const char* tmpdir = std::getenv("TEST_TMPDIR");
  const std::string path = std::string(tmpdir ? tmpdir : "/tmp") + "/trace.json";
  result->save(path);
  std::ifstream file(path);
  std::stringstream trace;
  trace << file.rdbuf();
  const std::string json = trace.str();

  const std::size_t cpu_ops = Count(json, "\"cat\": \"cpu_op\"");
  std::printf("Chrome trace: %zu bytes, %zu cpu_op events\n", json.size(), cpu_ops);
  EXPECT(json.find("\"traceEvents\"") != std::string::npos);
  EXPECT(cpu_ops > 0);
  EXPECT(json.find("aten::matmul") != std::string::npos);
  EXPECT(json.find("aten::relu") != std::string::npos);
  return test::Finish("kineto_profiler_test");
}
