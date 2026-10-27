# PHASE 5: Performance, Quantization, Profiling & Distributed Ecosystem (`@xnnpack`, `@fbgemm`, `@kineto`, `@gloo`)

> **Status:** In progress (follows [PHASE4.md](PHASE4.md)). Completes the `@libtorch` 2.14.1 feature set to match a full upstream production CMake build by packaging `@xnnpack`, `@fbgemm` (with `@asmjit`), `@kineto`, and `@gloo` as local BCR modules under `./modules/`, enabling INT8/FP16 quantized operators, Kineto CPU/GPU profiling, and `c10d` collective communications, and hardening the $6 \times 6$ CI matrix and sanitizers for upstreaming to the Bazel Central Registry.

## 0. Progress and Decisions (2026-10-26)

Every dependency is pinned to the commit that PyTorch 2.14.1's `third_party/` submodules pin (not to the releases of section 1):

| Module | Version | Notes |
| :--- | :--- | :--- |
| `kineto` | `0.0.0-20260805-15f4784` | `:kineto` (CPU) and `:kineto_cupti` (CUPTI, on `:cuda_enabled`), from `libkineto/libkineto_defs.bzl`; dynolog's IPC fabric on Linux. ROCm and XPU builds use `:kineto` (no rocprofiler-sdk or PTI). Patch: `<cstdlib>` for libc++. |
| `asmjit` | `0.0.0-20250628-a3199e8` | FBGEMM's submodule. Static (`ASMJIT_STATIC`). |
| `fbgemm` | `0.0.0-20260514-bf6dce3` | From `defs.bzl` (the repository's own `BUILD.bazel` only builds on linux-x86_64): AVX2/AVX-512 on x86_64, NEON on AArch64, and SVE2 and KleidiAI except on macOS. Patch: MinGW's `_aligned_malloc`. |
| `gloo` | `0.0.0-20260610-74cc005` | CPU only (Gloo's CUDA collectives are not built; `GLOO_USE_TORCH_DTYPES` only affects them): TCP transport on Linux, libuv elsewhere. Patch: `<exception>`, `<iterator>` for libc++. |
| `cpp-httplib` | `0.50.1` | Header-only, without TLS and zlib (the BCR's 0.46.0 depends on BoringSSL and zlib). |

`@libtorch` changes:

- `USE_KINETO` (and `HAS_CUPTI` with CUDA) and `USE_FBGEMM` (x86_64 and AArch64) are in `TORCH_CPU_LOCAL_DEFINES`: the profiler's and quantization's headers depend on them, so every `@libtorch` library agrees.
- `:distributed_enabled` (Linux only: upstream also builds it on Windows, but only with MSVC, as Gloo's Windows code and the BCR's libuv assume MSVC, not MinGW; and not on macOS by default): `torch_cpu` compiles `libtorch_distributed_base_sources` and `libtorch_distributed_extra_sources` (c10d's HashStore, distributed autograd, and RPC without the TensorPipe agent, as without `USE_TENSORPIPE`), with the public `USE_DISTRIBUTED` and `USE_C10D_GLOO`. TCPStore uses its own backend (upstream only enables `TORCH_USE_LIBUV` with TensorPipe).
- **XNNPACK is not built:** upstream 2.14.1 defaults `USE_XNNPACK` to `OFF` (`CMakeLists.txt`), so section 2.1 no longer matches upstream. `USE_PYTORCH_QNNPACK` and `USE_NNPACK` default to `ON` upstream but are not built yet (FBGEMM provides the quantized engine on both x86_64 and AArch64).

---

## 1. Local BCR Modules Added in Phase 5 (`./modules/`)

| Module | Version | Upstream Repository | License (`@rules_license`) | Purpose in `@libtorch` |
| :--- | :--- | :--- | :--- | :--- |
| `asmjit` | `0.0.0-20250501` | [`asmjit/asmjit`](https://github.com/asmjit/asmjit) | `Zlib` | x86_64/AArch64 JIT assembler required by `@fbgemm`. |
| `fbgemm` | `1.1.0` | [`pytorch/FBGEMM`](https://github.com/pytorch/FBGEMM) | `BSD-3-Clause` | Low-precision (INT8/FP16/BF16) matrix multiplication and embedding kernels on `x86_64` (`USE_FBGEMM=1`). |
| `xnnpack` | `0.0.0-20260101` | [`google/XNNPACK`](https://github.com/google/XNNPACK) | `BSD-3-Clause` | High-performance ARM64 NEON/SVE and x86 AVX2/AVX-512 operator micro-kernels (`USE_XNNPACK=1`). |
| `kineto` | `0.4.0` | [`pytorch/kineto`](https://github.com/pytorch/kineto) | `BSD-3-Clause` | `libkineto` CPU & GPU (CUDA CUPTI / ROCm roctracer / XPU PTI) profiler (`USE_KINETO=1`). |
| `gloo` | `0.5.0` | [`pytorch/gloo`](https://github.com/pytorch/gloo) | `BSD-3-Clause` | Collective communications (`AllReduce`, `Broadcast`, `AllGather`, `ReduceScatter`) over TCP/uv (`USE_DISTRIBUTED=1`, `USE_GLOO=1`). |
| `vulkan-memory-allocator` | `3.2.1` | [`GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator`](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | `MIT` | Header-only Vulkan memory allocator used by `ATen/native/vulkan`. |

---

## 2. Subsystem Integration Details

### 2.1 `@xnnpack` & `ATen/native/xnnpack/**` (`USE_XNNPACK=1`)

- PyTorch already ships `third_party/xnnpack_src_defs.bzl` and `third_party/xnnpack_wrapper_defs.bzl` in the upstream `pytorch@v2.14.1` tarball, which list the exact per-ISA source groups (`AARCH64_NEON_SRCS`, `X86_SSE2_SRCS`, `X86_AVX2_SRCS`, `X86_AVX512_SRCS`, `SCALAR_SRCS`).
- `modules/xnnpack/0.0.0-20260101/overlay/BUILD.bazel` adapts those source lists with `select()` on `@platforms//cpu:{x86_64,aarch64}` and depends on `@pthreadpool`, `@cpuinfo`, `@fp16`, `@fxdiv`, and `@psimd` from the upstream BCR.
- `@libtorch//:torch_cpu` defines `USE_XNNPACK=1` and compiles `aten/src/ATen/native/xnnpack/*.cpp`.

### 2.2 `@fbgemm` & Quantized Operators (`USE_FBGEMM=1`, `ATen/native/quantized/**`)

- Built on `x86_64` targets (`linux_amd64`, `macos_amd64`, `windows_amd64`).
- `modules/fbgemm/1.1.0/overlay/BUILD.bazel` compiles:
  - `:fbgemm_generic` (`src/*.cc`)
  - `:fbgemm_avx2` (`src/*Avx2.cc`, `copts = ["-mavx2", "-mfma", "-mf16c"]`)
  - `:fbgemm_avx512` (`src/*Avx512*.cc`, `copts = ["-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl"]`)
  - Depends on `@asmjit` and `@cpuinfo`.
- `@libtorch//:torch_cpu` defines `USE_FBGEMM=1` on `x86_64` and `USE_PYTORCH_QNNPACK=1` (internal `aten/src/ATen/native/quantized/cpu/qnnpack/**`) on both `x86_64` and `aarch64`.

### 2.3 `@kineto` Profiler (`USE_KINETO=1`, `torch/csrc/profiler/**`)

- `modules/kineto/0.4.0/overlay/BUILD.bazel` builds `libkineto` (`libkineto/src/*.cpp`):
  - Always compiles CPU trace/activity collection (`CpuTraceBuffer`, `ChromeTraceLogger`, `ActivityProfilerController`).
  - When `@rules_cuda//cuda:is_valid_toolchain_found` holds on CUDA targets, enables `CuptiActivityApi` against `@cuda//:cupti` (`<cupti.h>`).
  - Depends on `@fmt` and `@nlohmann_json` from BCR.
- `@libtorch//:torch_cpu` enables `USE_KINETO=1` and compiles `torch/csrc/profiler/kineto_shim.cpp` and `torch/csrc/profiler/collection.cpp`.

### 2.4 `@gloo` & Distributed Collectives (`USE_DISTRIBUTED=1`, `USE_C10D_GLOO=1`)

- `modules/gloo/0.5.0/overlay/BUILD.bazel` builds `@gloo//:gloo` on Linux and macOS (and Windows via Winsock/libuv):
  - Includes `gloo/allreduce*.cc`, `gloo/broadcast*.cc`, `gloo/gather*.cc`, `gloo/reduce*.cc`, `gloo/scatter*.cc`, `gloo/barrier*.cc`, and `gloo/transport/tcp/*.cc` (or `uv` on macOS/Windows).
- `@libtorch//:torch_cpu` compiles `torch/csrc/distributed/c10d/{Store,FileStore,TCPStore,PrefixStore,ProcessGroup,ProcessGroupGloo,Work,Backend}.cpp` (using `@cpp-httplib` from BCR for HTTP store support).

---

## 3. Warning Hygiene, Sanitizers & BCR Upstreaming Readiness

1. **Warning-Free `@libtorch` Builds:**
   - Just like `qt_bazel_workspace/.bazelrc`, third-party external dependencies (`.*external/(?!libtorch)[^/]+/.*@-w`) have warnings silenced so only `@libtorch`'s own diagnostics appear.
   - Any false-positive Clang 21 warnings on upstream PyTorch 2.14.1 (such as `-Wno-thread-safety-analysis`, `-Wno-deprecated-declarations`, `-Wno-unused-but-set-variable`) are scoped via `--per_file_copt=external/libtorch[^/]*[+]/.*@...` in `.bazelrc` and `LIBTORCH_COMMON_COPTS` in `libtorch_config.bzl`.
2. **Sanitizer Suites (`.bazelrc`):**
   - `--config=asan`, `--config=ubsan`, `--config=tsan`, `--config=msan`, and `--config=fuzztest` wired to `@llvm//config:{asan,ubsan,tsan,msan}`.
3. **BCR Presubmit & Licensing Compliance:**
   - Every module under `./modules/` carries a `@rules_license` `license()` target (`BSD-3-Clause` for `libtorch`, `sleef`, `fbgemm`, `xnnpack`, `kineto`, `gloo`; `BSD-3-Clause` / `GPL-2.0-only` for `ittapi`; `MIT` for `hip-stub`, `level-zero-stub`, `vulkan-memory-allocator`; `Zlib` for `asmjit`).
   - Every module includes a valid `presubmit.yml` and `metadata.json` verified by `tools/update_overlay_hashes.sh`.

---

## 4. Verification Criteria (Phase 5 Gate)

1. **`//tests/quantized_ops:quantized_ops_test`:**
   - Quantizes a float tensor to `torch::kQInt8` (`at::quantize_per_tensor`), executes `at::quantized_linear` and `at::quantized_max_pool2d` (exercising `@fbgemm` on `x86_64` and `qnnpack`/`@xnnpack` on `aarch64`), and dequantizes back to float.
2. **`//tests/kineto_profiler:kineto_profiler_test`:**
   - Runs a tensor workload inside a `torch::autograd::profiler::RecordProfile` / Kineto session, exports a Chrome trace JSON to a temporary file, and verifies non-empty `cpu_op` events in the parsed JSON.
3. **`//tests/distributed_gloo:distributed_gloo_test`:**
   - Spawns two in-process threads connected via `c10d::FileStore` (or `c10d::TCPStore` on `127.0.0.1`, matching `qt_bazel_workspace`'s in-process loopback tests), initializes a 2-rank `c10d::ProcessGroupGloo`, and verifies an `allreduce` sum across ranks `0` and `1`.
4. **Full 36-Job CI Matrix Green:**
   - All 6 native build+test jobs and all 30 cross-compilation jobs pass on Bazel `9.x` and `last_rc`.

