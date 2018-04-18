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

tf_custom_op_library(
    name = "direct_sparse_conv_kd.so",

    srcs =  [
              "direct_sparse_conversions.cc",
              "direct_sparse_conv_kd.cc",
              "direct_sparse_pooling_kd.cc",
              "direct_sparse_to_dense.cc",
              "evaluate_confusion_matrix.cc",
              "direct_sparse_cwise_biased_reg.cc",
              "direct_sparse_concat.cc",
            ],

    gpu_srcs = [
              "direct_sparse_cuda_helpers_gpu.h",
              "direct_sparse_conversions_gpu.h",
              "direct_sparse_conversions_gpu.cu.cc",
              "direct_sparse_conv_kd_gpu.h",
              "direct_sparse_conv_kd_gpu.cu.cc",
              "direct_sparse_pooling_kd_gpu.h",
              "direct_sparse_pooling_kd_gpu.cu.cc",
              "direct_sparse_to_dense_gpu.h",
              "direct_sparse_to_dense_gpu.cu.cc",
              "direct_sparse_cwise_biased_reg_gpu.h",
              "direct_sparse_cwise_biased_reg_gpu.cu.cc",
              "direct_sparse_concat_gpu.h",
              "direct_sparse_concat_gpu.cu.cc",
    ],

    deps = [
              "@cub_archive//:cub",
    ],
)

