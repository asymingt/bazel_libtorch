// Smoke test for @libtorch//:c10 (and :headeronly): device types, the CPU
// allocator, small containers, scalar types and PrivateUse1 registration.

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/StringUtil.h>
#include <torch/headeronly/macros/Export.h>
#include <torch/headeronly/version.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,     \
                   __LINE__, #cond);                                  \
      ++failures;                                                     \
    }                                                                 \
  } while (false)

void TestDeviceTypes() {
  static_assert(c10::COMPILE_TIME_MAX_DEVICE_TYPES == 21);

  const std::vector<std::pair<c10::DeviceType, std::string>> expected = {
      {c10::DeviceType::CPU, "CPU"},
      {c10::DeviceType::CUDA, "CUDA"},
      {c10::DeviceType::MKLDNN, "MKLDNN"},
      {c10::DeviceType::OPENGL, "OPENGL"},
      {c10::DeviceType::OPENCL, "OPENCL"},
      {c10::DeviceType::IDEEP, "IDEEP"},
      {c10::DeviceType::HIP, "HIP"},
      {c10::DeviceType::FPGA, "FPGA"},
      {c10::DeviceType::MAIA, "MAIA"},
      {c10::DeviceType::XLA, "XLA"},
      {c10::DeviceType::Vulkan, "VULKAN"},
      {c10::DeviceType::Metal, "METAL"},
      {c10::DeviceType::XPU, "XPU"},
      {c10::DeviceType::MPS, "MPS"},
      {c10::DeviceType::Meta, "META"},
      {c10::DeviceType::HPU, "HPU"},
      {c10::DeviceType::VE, "VE"},
      {c10::DeviceType::Lazy, "LAZY"},
      {c10::DeviceType::IPU, "IPU"},
      {c10::DeviceType::MTIA, "MTIA"},
      {c10::DeviceType::PrivateUse1, "PRIVATEUSEONE"},
  };
  CHECK(expected.size() == c10::COMPILE_TIME_MAX_DEVICE_TYPES);
  for (size_t i = 0; i < expected.size(); ++i) {
    const auto& [type, name] = expected[i];
    CHECK(static_cast<size_t>(type) == i);
    CHECK(c10::isValidDeviceType(type));
    CHECK(c10::DeviceTypeName(type) == name);
  }

  const c10::Device cuda1("cuda:1");
  CHECK(cuda1.is_cuda());
  CHECK(cuda1.index() == 1);
  CHECK(cuda1.str() == "cuda:1");
  CHECK(c10::Device("cpu").is_cpu());
  CHECK(c10::Device("mps").is_mps());
  CHECK(c10::Device("meta").is_meta());
}

void TestCPUAllocator() {
  c10::Allocator* allocator = c10::GetCPUAllocator();
  CHECK(allocator != nullptr);
  constexpr size_t kBytes = 1024;
  c10::DataPtr data = allocator->allocate(kBytes);
  CHECK(data.get() != nullptr);
  CHECK(data.device().is_cpu());
  CHECK(reinterpret_cast<uintptr_t>(data.get()) % c10::gAlignment == 0);
  std::memset(data.get(), 0x5a, kBytes);
  CHECK(static_cast<unsigned char*>(data.get())[kBytes - 1] == 0x5a);
}

void TestContainers() {
  c10::SmallVector<int64_t, 4> sizes = {2, 3, 4};
  sizes.push_back(5);
  sizes.push_back(6);  // Spills to the heap.
  CHECK(sizes.size() == 5);
  CHECK(sizes[4] == 6);

  const c10::IntArrayRef ref(sizes);
  CHECK(ref.size() == 5);
  CHECK(ref.front() == 2);
  CHECK(ref.back() == 6);
  CHECK(ref.slice(1, 2).vec() == std::vector<int64_t>({3, 4}));
  CHECK(c10::str(ref) == "[2, 3, 4, 5, 6]");
}

void TestScalarTypes() {
  CHECK(c10::elementSize(c10::ScalarType::Float) == 4);
  CHECK(c10::elementSize(c10::ScalarType::Half) == 2);
  CHECK(c10::elementSize(c10::ScalarType::BFloat16) == 2);
  CHECK(c10::elementSize(c10::ScalarType::Float8_e4m3fn) == 1);
  CHECK(c10::isFloatingType(c10::ScalarType::BFloat16));
  CHECK(c10::isFloat8Type(c10::ScalarType::Float8_e4m3fn));
  CHECK(std::string(c10::toString(c10::ScalarType::Half)) == "Half");
  CHECK(c10::promoteTypes(c10::ScalarType::Half, c10::ScalarType::Float) ==
        c10::ScalarType::Float);

  const c10::Half half(1.5f);
  CHECK(static_cast<float>(half) == 1.5f);
  const c10::BFloat16 bfloat16(2.0f);
  CHECK(static_cast<float>(bfloat16) == 2.0f);
  const c10::Float8_e4m3fn float8(0.5f);
  CHECK(static_cast<float>(float8) == 0.5f);
}

void TestPrivateUse1() {
  CHECK(!c10::is_privateuse1_backend_registered());
  c10::register_privateuse1_backend("custom_npu");
  CHECK(c10::is_privateuse1_backend_registered());
  CHECK(c10::get_privateuse1_backend() == "custom_npu");
  CHECK(c10::DeviceTypeName(c10::DeviceType::PrivateUse1, true) ==
        "custom_npu");
  const c10::Device npu("custom_npu:0");
  CHECK(npu.is_privateuseone());
  CHECK(npu.index() == 0);
}

}  // namespace

int main() {
  std::printf("LibTorch %s\n", TORCH_VERSION);
  TestDeviceTypes();
  TestCPUAllocator();
  TestContainers();
  TestScalarTypes();
  TestPrivateUse1();
  if (failures != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("All c10 checks passed\n");
  return 0;
}
