load("//tensorflow:tensorflow.bzl", "tf_custom_op_library_additional_deps", "tf_copts", "clean_dep")

def tf_custom_op_library_flags(name, srcs=[], gpu_srcs=[], deps=[]):
  deps = deps + tf_custom_op_library_additional_deps()

  native.cc_binary(
      name=name,
      srcs=srcs,
      deps=deps,
      copts=tf_copts() + ["-fopenmp", "-fexceptions", "-D_GLIBCXX_USE_CXX11_ABI=0"],
      linkshared=1,
      linkopts= [
              "-lm", "-L/usr/local/lib", "-lgomp", "-lcityhash"
          ],
     )

