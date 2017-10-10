load("//tensorflow:tensorflow.bzl", "tf_custom_op_library_additional_deps", "tf_copts", "clean_dep", "if_cuda", "cuda_default_copts", "check_deps")


def _cuda_copts_():
  """Gets the appropriate set of copts for (maybe) CUDA compilation.

    If we're doing CUDA compilation, returns copts for our particular CUDA
    compiler.  If we're not doing CUDA compilation, returns an empty list.

    """
  return cuda_default_copts() + select({
      "//conditions:default": [],
      "@local_config_cuda//cuda:using_nvcc": ([
          "-nvcc_options=relaxed-constexpr",
          "-nvcc_options=ftz=true",
      ]),
      "@local_config_cuda//cuda:using_clang": ([
          "-fcuda-flush-denormals-to-zero",
      ]),
  })



def tf_custom_op_library_flags(name, srcs=[], gpu_srcs=[], deps=[], cflags=[]):
  cuda_deps = [
      clean_dep("//tensorflow/core:stream_executor_headers_lib"),
      "@local_config_cuda//cuda:cudart_static",
  ]
  deps = deps + tf_custom_op_library_additional_deps()
  if gpu_srcs:
    basename = name.split(".")[0]
    native.cc_library(
        name=basename + "_gpu",
        srcs=gpu_srcs,
        copts=_cuda_copts_() +  ["-Xptxas='-v'"],
        deps=deps + if_cuda(cuda_deps))
    cuda_deps.extend([":" + basename + "_gpu"])

  check_deps(
      name=name + "_check_deps",
      deps=deps + if_cuda(cuda_deps),
      disallowed_deps=[
          clean_dep("//tensorflow/core:framework"),
          clean_dep("//tensorflow/core:lib")
      ])

  native.cc_binary(
      name=name,
      srcs=srcs,
      deps=deps + if_cuda(cuda_deps),
      copts=tf_copts() + ["-fopenmp", "-fexceptions"],
      linkshared=1,
      linkopts= cflags,
     )





