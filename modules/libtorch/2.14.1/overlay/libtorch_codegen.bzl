"""Build-time code generation for @libtorch.

Runs upstream's generators as Bazel actions on the exec platform, with the
hermetic Python toolchain:

- torchgen.gen (ATen operators, dispatcher registrations, AOTInductor shims),
- tools/setup_helpers/generate_code.py (autograd, TorchScript tracing and the
  lazy tensor TorchScript backend).

Nothing generated is checked in. Bazel needs the outputs up front, so they are
listed below, without per-operator headers (upstream's
AT_PER_OPERATOR_HEADERS mode, whose ~3,000 file names depend on
native_functions.yaml). `python -m torchgen.gen ... -o <file>` prints the list
that torchgen writes, for when it changes.
"""

# torchgen.gen outputs, relative to aten/src/ATen.
ATEN_GENERATED_HEADERS = [
    "CPUFunctions.h",
    "CPUFunctions_inl.h",
    "CompositeExplicitAutogradFunctions.h",
    "CompositeExplicitAutogradFunctions_inl.h",
    "CompositeExplicitAutogradNonFunctionalFunctions.h",
    "CompositeExplicitAutogradNonFunctionalFunctions_inl.h",
    "CompositeImplicitAutogradFunctions.h",
    "CompositeImplicitAutogradFunctions_inl.h",
    "CompositeImplicitAutogradNestedTensorFunctions.h",
    "CompositeImplicitAutogradNestedTensorFunctions_inl.h",
    "FunctionalInverses.h",
    "Functions.h",
    "MetaFunctions.h",
    "MetaFunctions_inl.h",
    "MethodOperators.h",
    "NativeFunctions.h",
    "NativeMetaFunctions.h",
    "Operators.h",
    "RedispatchFunctions.h",
    "RegistrationDeclarations.h",
    "ViewMetaClasses.h",
    "VmapGeneratedPlumbing.h",
    "core/TensorBody.h",
    "core/aten_interned_strings.h",
]

ATEN_GENERATED_SOURCES = [
    "CompositeViewCopyKernels.cpp",
    "Functions.cpp",
    "Operators_0.cpp",
    "Operators_1.cpp",
    "Operators_2.cpp",
    "Operators_3.cpp",
    "Operators_4.cpp",
    "RegisterBackendSelect.cpp",
    "RegisterCPU_0.cpp",
    "RegisterCPU_1.cpp",
    "RegisterCPU_2.cpp",
    "RegisterCPU_3.cpp",
    "RegisterCompositeExplicitAutogradNonFunctional_0.cpp",
    "RegisterCompositeExplicitAutograd_0.cpp",
    "RegisterCompositeImplicitAutogradNestedTensor_0.cpp",
    "RegisterCompositeImplicitAutograd_0.cpp",
    "RegisterFunctionalization_0.cpp",
    "RegisterFunctionalization_1.cpp",
    "RegisterFunctionalization_2.cpp",
    "RegisterFunctionalization_3.cpp",
    "RegisterMeta_0.cpp",
    "RegisterMkldnnCPU_0.cpp",
    "RegisterNestedTensorCPU_0.cpp",
    "RegisterNestedTensorHPU_0.cpp",
    "RegisterNestedTensorMeta_0.cpp",
    "RegisterQuantizedCPU_0.cpp",
    "RegisterQuantizedMeta_0.cpp",
    "RegisterSchema.cpp",
    "RegisterSparseCPU_0.cpp",
    "RegisterSparseCsrCPU_0.cpp",
    "RegisterSparseCsrMeta_0.cpp",
    "RegisterSparseMeta_0.cpp",
    "RegisterZeroTensor_0.cpp",
    "UfuncCPU_add.cpp",
    "ViewMetaClasses.cpp",
    "core/ATenOpList.cpp",
    "core/TensorMethods.cpp",
]

# Compiled once per CPU capability, like aten/src/ATen/native/cpu/*.cpp.
ATEN_GENERATED_CPU_KERNEL_SOURCES = [
    "UfuncCPUKernel_add.cpp",
]

ATEN_GENERATED_CUDA_HEADERS = [
    "CUDAFunctions.h",
    "CUDAFunctions_inl.h",
]

ATEN_GENERATED_CUDA_SOURCES = [
    "RegisterCUDA_0.cpp",
    "RegisterNestedTensorCUDA_0.cpp",
    "RegisterQuantizedCUDA_0.cpp",
    "RegisterSparseCUDA_0.cpp",
    "RegisterSparseCsrCUDA_0.cpp",
]

ATEN_GENERATED_CUDA_KERNEL_SOURCES = [
    "UfuncCUDA_add.cu",
]

# torchgen.gen outputs elsewhere in the tree.
TORCHGEN_OTHER_HEADERS = [
    "torch/headeronly/core/enum_tag.h",
]

AOTI_GENERATED_SOURCES = [
    "torch/csrc/inductor/aoti_torch/generated/c_shim_aten.cpp",
    "torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp",
]

AOTI_GENERATED_CUDA_SOURCES = [
    "torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.cpp",
]

# tools/setup_helpers/generate_code.py --subset libtorch --gen-lazy-ts-backend
# outputs.
AUTOGRAD_GENERATED_HEADERS = [
    "torch/csrc/autograd/generated/Functions.h",
    "torch/csrc/autograd/generated/VariableType.h",
    "torch/csrc/autograd/generated/ViewFuncs.h",
    "torch/csrc/autograd/generated/variable_factories.h",
    "torch/csrc/lazy/generated/LazyIr.h",
    "torch/csrc/lazy/generated/LazyNativeFunctions.h",
    "torch/csrc/lazy/generated/LazyNonNativeIr.h",
]

AUTOGRAD_GENERATED_SOURCES = [
    "torch/csrc/autograd/generated/ADInplaceOrViewType_0.cpp",
    "torch/csrc/autograd/generated/ADInplaceOrViewType_1.cpp",
    "torch/csrc/autograd/generated/Functions.cpp",
    "torch/csrc/autograd/generated/ViewFuncs.cpp",
    "torch/csrc/lazy/generated/LazyNativeFunctions.cpp",
    "torch/csrc/lazy/generated/RegisterAutogradLazy.cpp",
    "torch/csrc/lazy/generated/RegisterLazy.cpp",
] + [
    "torch/csrc/autograd/generated/{}_{}.cpp".format(kind, shard)
    for kind in ["TraceType", "VariableType"]
    for shard in range(10)
]

def _codegen_impl(ctx):
    outputs = {
        group: [ctx.actions.declare_file(path) for path in paths]
        for group, paths in ctx.attr.output_groups.items()
    }
    source_root = ctx.label.workspace_root or "."
    bin_root = ctx.bin_dir.path + ("/" + ctx.label.workspace_root if ctx.label.workspace_root else "")
    all_outputs = [f for files in outputs.values() for f in files]
    outputs_file = ctx.actions.declare_file(ctx.label.name + ".outputs")
    ctx.actions.write(outputs_file, "\n".join([f.path for f in all_outputs]) + "\n")
    arguments = [
        ctx.attr.module,
        source_root,
        bin_root,
        outputs_file.path,
    ] + ctx.attr.arguments
    ctx.actions.run(
        executable = ctx.executable._codegen,
        arguments = arguments,
        inputs = ctx.files.srcs + [outputs_file],
        outputs = all_outputs,
        mnemonic = ctx.attr.mnemonic,
        progress_message = "Generating %{label}",
    )
    return [
        DefaultInfo(files = depset(all_outputs)),
        OutputGroupInfo(**{group: depset(files) for group, files in outputs.items()}),
    ]

_codegen = rule(
    implementation = _codegen_impl,
    attrs = {
        "arguments": attr.string_list(
            doc = "Arguments of the module. {bin} is the repository's root in the bin directory.",
        ),
        "mnemonic": attr.string(mandatory = True),
        "module": attr.string(mandatory = True, doc = "The generator's Python module."),
        "output_groups": attr.string_list_dict(
            mandatory = True,
            doc = "Outputs (relative to the package) by output group.",
        ),
        "srcs": attr.label_list(allow_files = True, doc = "Inputs of the generator."),
        "_codegen": attr.label(
            default = "//tools:codegen",
            executable = True,
            cfg = "exec",
        ),
    },
)

def _aten(paths):
    return ["aten/src/ATen/" + path for path in paths]

def libtorch_aten_codegen(name):
    """Runs torchgen.gen. Output groups: headers, sources, cpu_kernel_sources,
    cuda_headers, cuda_sources, cuda_kernel_sources, aoti_sources and
    aoti_cuda_sources.

    Args:
        name: Name of the target.
    """
    _codegen(
        name = name,
        srcs = native.glob([
            "aten/src/ATen/native/*.yaml",
            "aten/src/ATen/templates/**",
        ]),
        arguments = [
            "--source-path=aten/src/ATen",
            "--install-dir={bin}/aten/src/ATen",
            "--aoti-install-dir={bin}/torch/csrc/inductor/aoti_torch/generated",
            "--headeronly-install-dir={bin}/torch/headeronly/core",
        ],
        mnemonic = "TorchGen",
        module = "torchgen.gen",
        output_groups = {
            "aoti_cuda_sources": AOTI_GENERATED_CUDA_SOURCES,
            "aoti_sources": AOTI_GENERATED_SOURCES,
            "cpu_kernel_sources": _aten(ATEN_GENERATED_CPU_KERNEL_SOURCES),
            "cuda_headers": _aten(ATEN_GENERATED_CUDA_HEADERS),
            "cuda_kernel_sources": _aten(ATEN_GENERATED_CUDA_KERNEL_SOURCES),
            "cuda_sources": _aten(ATEN_GENERATED_CUDA_SOURCES),
            "headers": _aten(ATEN_GENERATED_HEADERS) + TORCHGEN_OTHER_HEADERS,
            "sources": _aten(ATEN_GENERATED_SOURCES),
        },
    )

def libtorch_autograd_codegen(name):
    """Runs tools/setup_helpers/generate_code.py. Output groups: headers and
    sources.

    Args:
        name: Name of the target.
    """
    _codegen(
        name = name,
        srcs = native.glob([
            "aten/src/ATen/native/*.yaml",
            "aten/src/ATen/templates/**",
            "torch/csrc/lazy/**/*.h",
            "torch/csrc/lazy/ts_backend/ts_native_functions.cpp",
        ]),
        arguments = [
            "--gen-dir={bin}",
            "--native-functions-path=aten/src/ATen/native/native_functions.yaml",
            "--tags-path=aten/src/ATen/native/tags.yaml",
            "--subset=libtorch",
            "--gen-lazy-ts-backend",
        ],
        mnemonic = "AutogradGen",
        module = "tools.setup_helpers.generate_code",
        output_groups = {
            "headers": AUTOGRAD_GENERATED_HEADERS,
            "sources": AUTOGRAD_GENERATED_SOURCES,
        },
    )
