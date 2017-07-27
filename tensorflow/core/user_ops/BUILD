package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)


load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_py_wrap_cc")
load("//tensorflow/core/user_ops:custom_bzl.bzl", "tf_custom_op_library_flags")

tf_custom_op_library_flags(
    name = "direct_sparse_conv_kd.so",

    srcs =  [
              "sparse_tensor_sparse_kernel_dense_conv_kd.h",
              "sparse_tensor_sparse_kernel_dense_conv_kd_grad.h",
              "sparse_tensor_sparse_kernel_dense_conv_kd.cc",
              "sparse_tensor_sparse_kernel_dense_conv_kd_grad.cc",
              "sparse_tensor_dense_relu_op.cc",
              "sparse_tensor_dense_relu_op_functor.h",
              "sparse_tensor_dense_relu_op.h",
              "indexing.h",
              "hash_map.h",
              "sparse_tensor_pooling.cc",
              "sparse_tensor_pooling_grad.cc",
              "sparse_tensor_pooling.h",
              "direct_sparse_to_dense_op.cc",
              "sparse_tensor_filter_zero_entries.cc",
              "evaluate_confusion_matrix.cc",
              "direct_sparse_conv_kd.cc",
            ],

    gpu_srcs = [
              "tf_cudpp_bindings_gpu.h",
              "direct_sparse_conv_kd_gpu.h",
              "direct_sparse_conv_kd_gpu.cu.cc",
              "direct_sparse_cuda_helpers_gpu.h",
    ],

    deps = [
              "@cub_archive//:cub",
    ],

    cflags = [
      "-lm", "-L/usr/local/lib", "-lgomp", "-lcityhash", "-lcudpp", "-lcudpp_hash",
    ],
)

py_library(
    name = "sparse_nn_ops",
    srcs = [
              "sparse_ops.py",
              "sparse_module",
              "sparse_tools.py",
           ],
    data = [":sparse_tensor_dense_conv_3d.so"],
    srcs_version = "PY2AND3",
)

tf_py_test(
    name = "sparse_tensor_dense_conv_3d_test",
    size = "small",
    srcs = [
              "tests/sparse_tensor_dense_conv_3d_test.py",
              "tests/sparse_tools.py",
              "tests/sparse_tensor_dense_relu_op_test.py",   
           ],
    additional_deps = ["//tensorflow:tensorflow_py"],
)


