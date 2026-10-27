# Progress Report (as of 2026-10-27)

This report summarizes what has been built across all five phases of the plan in [DESIGN.md](DESIGN.md), what deviates from that plan and why, and what is still open. The phase documents keep their original designs, and Phases 4 and 5 also open with a dated "Progress and Decisions" section. This file is the single place to see the overall state.

## 1. At a Glance

| Phase | Scope | State |
| :--- | :--- | :--- |
| [1](PHASE1.md) | Registry, hermetic toolchains, `:headeronly`, `:c10` | Done |
| [2](PHASE2.md) | Hermetic `torchgen`, `:torch_cpu`, `:torch` | Done |
| [3](PHASE3.md) | NVIDIA CUDA (`:c10_cuda`, `:torch_cuda`) | Done on Linux; not on Windows |
| [4](PHASE4.md) | Vulkan, Apple MPS, AMD ROCm, Intel XPU | Done; XPU is runtime-only |
| [5](PHASE5.md) | Kineto, FBGEMM, Gloo/c10d (XNNPACK dropped) | Gate tests pass; QNNPACK, NNPACK and section 3 (hygiene, sanitizers) open |

**Build and test status:**

- Linux x86_64 (native): all 14 tests applicable to the default configuration pass; 3 are skipped because they need another configuration.
- `--config=rocm` and `--config=xpu`: their dispatch tests pass.
- Other five targets: `//...` cross-builds for each of `linux_arm64`, `macos_arm64`, `macos_amd64`, `windows_amd64` and `windows_arm64`.
- These results come from local builds on BuildBuddy remote execution; the 36-job CI matrix has not run on Phases 4 and 5 yet.
- Phases 4 and 5 are not committed yet (the last commit, `367be2d`, is mid-Phase 4).

## 2. How a Build Is Configured

One discrete GPU backend is built per build, chosen with `--@libtorch//:gpu`. Apple MPS and Vulkan sit alongside it, each with its own on/off flag:

| Setting | Values | Default | Notes |
| :--- | :--- | :--- | :--- |
| `--@libtorch//:gpu` | `cuda`, `rocm`, `xpu`, `none` | `cuda` | Mutually exclusive, as upstream (ROCm reuses the CUDA backend through hipify). `--config=rocm` and `--config=xpu` select the others. |
| `--@libtorch//:rocm_archs` | AMD GPU list | `gfx90a,gfx942,gfx1100` | Device code is compiled for each. |
| `--@rules_cuda//cuda:archs` | NVIDIA GPU list | `sm_70` … `sm_90` + `compute_90` | `.bazelrc`; override locally with your GPU's architecture. |
| `--@libtorch//:enable_mps` | bool | `True` | Built on `macos_arm64` only. |
| `--@libtorch//:enable_vulkan` | bool | `True` | Built on Linux and Windows. |

Depend on the resolved settings (`@libtorch//:cuda_enabled`, `:rocm_enabled`, `:xpu_enabled`, `:mps_enabled`, `:vulkan_enabled`, `:distributed_enabled`), not on the flags.

**Where each backend is built:**

| | CUDA | ROCm | XPU | Vulkan | MPS | FBGEMM | Kineto | c10d + Gloo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| linux_amd64 | ✓ | ✓ | ✓ (stub) | ✓ | | ✓ | ✓ (CUPTI with CUDA) | ✓ |
| linux_arm64 | ✓ | | | ✓ | | ✓ | ✓ | ✓ |
| macos_arm64 | | | | | ✓ | ✓ | ✓ | |
| macos_amd64 | | | | | | ✓ | ✓ | |
| windows_amd64 | | | | ✓ | | ✓ | ✓ | |
| windows_arm64 | | | | ✓ | | ✓ | ✓ | |

## 3. Phase by Phase

### Phase 1: Foundation

- **Local registry:** `bazel_registry.json` plus `modules/`, with every overlay hash-pinned (`tools/update_overlay_hashes.sh`). One CI matrix covers 6 hosts × 6 targets.
- **Hermetic toolchain:** `@llvm` 0.8.25 (Clang 23, libc++) with 11 patches (`patches/llvm/`), plus `@rules_cuda` 0.3.0 with 2 patches (`patches/rules_cuda/`).
- **BCR module patches:** `cpuinfo` (Windows-on-Arm header) and `gflags` (choose Windows by target platform).
- **Targets:** `@libtorch//:headeronly`, `:c10` and `:c10_shared`.
  - `glog` is pinned at a patched 0.6.0 (`0.6.0.bcr.0`), because c10 needs glog ≤ 0.6.
  - MinGW-compatibility headers let the MSVC-oriented sources build for Windows with Clang.
- **Tests:** `c10_smoke`, `hello_world`.

### Phase 2: CPU LibTorch

- **Code generation:** `torchgen` and `tools/autograd` run hermetically as Bazel actions (`libtorch_codegen.bzl`).
- **CPU kernels:** compiled once per CPU capability: AVX2 and AVX-512 on x86_64, SVE128 and SVE256 on Linux arm64.
- **Targets:** `:torch_cpu` and `:torch`.
- **Companion modules:** `sleef` 3.8.0, `ittapi` 3.26.3, `flatbuffers` 24.12.23.
- **CPU configuration:** ATen is built without BLAS/LAPACK, MKL or oneDNN. pocketfft provides FFTs, and ATen's native thread pool replaces OpenMP. So CPU linear algebra raises at runtime.
- **Tests:** `tensor_math`, `autograd_nn`, `torchscript_jit`, `accelerator_hooks`.

### Phase 3: NVIDIA CUDA

- **Compilation:** `:c10_cuda`, `:aten_cuda_kernels` and `:torch_cuda`, compiled by Clang (not nvcc) against the hash-pinned CUDA 12.8.1 redistributables.
- **Attention:** the CUTLASS-based memory-efficient attention kernels are built (`cutlass` 4.6.1); FlashAttention and cuDNN are not.
- **Platforms:** Linux x86_64 and Linux arm64 (sbsa). Windows is not built, because the CUDA headers assume MSVC whenever `_WIN32` is defined.
- **Test:** `cuda_dispatch`, which runs cuBLAS, cuSOLVER, cuFFT and memory-efficient attention against the CPU when a GPU is present (verified on an RTX 4060).

### Phase 4: Multi-vendor GPUs

| Backend | Result |
| :--- | :--- |
| Vulkan | Compute shaders compiled to SPIR-V by `glslang` 16.6.0. The loader is loaded at runtime by `volk`, and memory is managed by `vulkan-memory-allocator` 3.3.0. `vulkan_dispatch` passes on a real GPU. |
| Apple MPS | Objective-C++ sources and the Metal shaders (compiled at runtime) in `:torch_cpu`. `mps_dispatch` cross-builds for macos_arm64. |
| AMD ROCm | `:c10_hip`, `:torch_hip_kernels` and `:torch_hip`, built from the hipified CUDA backend. `hip_smoke`, `rocm_libraries` and `hip_dispatch` pass without an AMD GPU (0 devices). |
| Intel XPU | `:c10_xpu` and `:torch_xpu` (runtime only). `xpu_dispatch` passes and always reports 0 devices. |

**ROCm in more detail:**

- **Version:** ROCm 7.14.1, pinned to the `therock-7.14.1` source tags (not 6.3.2), because PyTorch 2.14.1 has `ROCM_VERSION >= 71400` code paths.
- **Toolchain:** upstream Clang 23 compiles the HIP device code. Three modules support it:
  - `rocm-device-libs`: AMDGPU bitcode compiled from source.
  - `hip-stub`: HIP headers, plus `:hip_sdk`, a ROCm directory for Clang that includes the offload tools built from `@llvm-project`.
  - `rocm-libraries-stub`: rocBLAS, rocSOLVER, hipBLAS(Lt), hipSOLVER, hipSPARSE, hipFFT and MIOpen headers, plus the header-only rocPRIM, hipCUB and rocThrust.
- **Library shims:** each ROCm library is a generated shim with the real SONAME and symbol versions. On first call it loads the real library from `ROCM_PATH`, `/opt/rocm` or the library path. Without ROCm it reports no devices or an error code instead.
- **Hipify:** one action (`tools/libtorch_hipify.py`) produces a header tree, which takes precedence over the original headers through `-I`, and source trees for `c10_hip` and `torch_hip`.
- **libc++ differences:** `rocm/libtorch_hip_compat.h` is force-included to cover what upstream's libstdc++ build gets for free.

**XPU in more detail:**

- **Runtime only:** the operators live in the separate `torch-xpu-ops` repository and need Intel's SYCL compiler.
- **SYCL is a stub:** Intel's `libsycl` is built against libstdc++, whose C++ ABI libc++ code cannot call. So `sycl-stub` (DPC++ 7.2.0 headers plus a `libsycl.so.9` stub) never loads it.
- **Other modules:** `opencl-headers` 2025.07.22 and `level-zero` 1.35.0 (headers only).

### Phase 5: Profiling, quantization, distributed

| Subsystem | Modules | Test |
| :--- | :--- | :--- |
| Kineto profiler (`USE_KINETO`) | `kineto` (CPU everywhere, plus CUPTI with CUDA) | `kineto_profiler`: a Chrome trace with `cpu_op` events |
| FBGEMM quantized operators (`USE_FBGEMM`) | `fbgemm`, `asmjit`: AVX2/AVX-512 on x86_64, NEON on Arm, SVE2/KleidiAI except on macOS | `quantized_ops`: INT8 linear and max pooling against float |
| c10d with Gloo (`USE_DISTRIBUTED`, `USE_C10D_GLOO`) | `gloo` (CPU, TCP transport), `cpp-httplib` 0.50.1 | `distributed_gloo`: 2-rank allreduce |

**Notes:**

- **Versions:** every dependency is pinned to the commit of PyTorch 2.14.1's `third_party/` submodule.
- **Distributed scope:** distributed is built on Linux only. Upstream also builds it on Windows, but only with MSVC.
- **No TensorPipe:** RPC builds without its TensorPipe agent, so `USE_TENSORPIPE` is off.

## 4. Deviations From the Original Plan

| Plan | What was done | Why |
| :--- | :--- | :--- |
| Separate `enable_cuda`, `enable_rocm` and `enable_xpu` flags | One `--@libtorch//:gpu` flag | The backends are mutually exclusive upstream. |
| ROCm 6.3.2, with `@rocm_device_libs` + `@hip_stub` | ROCm 7.14.1, as `rocm-device-libs` + `hip-stub` + `rocm-libraries-stub` | PyTorch needs 7.14; each module has a single source archive. |
| Link-time stub libraries | Lazily-forwarding shims | One binary both runs on AMD GPUs and starts without ROCm. |
| XPU with oneDNN operators, forwarding to Intel's runtime | Runtime only, with a stub that never loads Intel's runtime | No `-fsycl` compiler; libstdc++/libc++ ABI mismatch. |
| XNNPACK in Phase 5 | Not built | Upstream 2.14.1 defaults `USE_XNNPACK` to off. |
| Releases as versions (e.g. `fbgemm` 1.1.0, `kineto` 0.4.0) | PyTorch's submodule commits | These are what PyTorch 2.14.1 is tested with. |
| Distributed on Linux and Windows | Linux only | Gloo and the BCR's libuv assume MSVC on Windows; our toolchain targets MinGW. |
| Kineto with ROCm and XPU tracing | CPU-only Kineto for ROCm and XPU | rocprofiler-sdk and Intel's PTI are not available. |

## 5. Upstream Issues Found

These are worth reporting to their projects:

- **`@llvm`:** the patched `llvm-link` (`llvm-link-multicall.patch`) passes `argv[0]` to its option parser, so it reads its own binary as an input file. The prebuilt multicall binary's `llvm link` dispatch fails too.
- **`@llvm`:** the minimal toolchain has no `clang-linker-wrapper`, `clang-offload-bundler` or `llvm-offload-binary`, which HIP needs. `hip-stub` builds them from source.
- **XNNPACK:** its `build_config` selects on legacy `--cpu` values, so it can't cross-compile with platform-based toolchain resolution.
- **Missing includes under libc++:** Kineto, Gloo and rocThrust rely on headers that libstdc++ includes transitively and libc++ does not. Each is fixed by a small patch here.
- **FBGEMM:** its repository's `BUILD.bazel` only works on linux-x86_64.

## 6. Open Work

1. **Commit Phases 4 and 5**, then run the full 36-job CI matrix, which includes the new ROCm and XPU jobs.
2. **Phase 5 gaps:**
   - QNNPACK and NNPACK, both on by default upstream.
   - Section 3: warning hygiene, sanitizer suites, and a BCR presubmit check of every module.
3. **README:** document the new modules, `--config=rocm` and `--config=xpu`, and the platform matrix above.
4. **Hardware validation:** the ROCm path has never run on an AMD GPU. CUDA (RTX 4060) and Vulkan were tested on real hardware here.
5. **Platform gaps:**
   - CUDA on Windows (blocked by the CUDA headers).
   - ROCm and XPU on Windows (would need DLL shims).
   - Distributed on Windows (would need MinGW support in Gloo and libuv).
6. **Known limitations to keep in mind:**
   - CPU linear algebra raises at runtime (no LAPACK).
   - On ROCm, TunableOp has no hipBLASLt candidates (libc++/libstdc++ ABI).
   - XPU never finds a device.
   - Stale status lines: PHASE1–3.md still say "Ready for execution" or "Planned".
