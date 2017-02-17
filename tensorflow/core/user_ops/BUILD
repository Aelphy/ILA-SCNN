load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "sparse_tensor_dense_conv_3d.so",

    srcs =  [
              "sparse_tensor_dense_conv_3d.cc",
              "sparse_tensor_sparse_kernel_dense_conv_3d.h",
              "sparse_tensor_sparse_kernel_dense_conv_3d.cc"
            ],
)

tf_py_test(
    name = "sparse_tensor_dense_conv_3d_test",
    size = "small",
    srcs = ["sparse_tensor_dense_conv_3d_test.py"],
    additional_deps = ["//tensorflow:tensorflow_py"],
)
