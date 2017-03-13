load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "sparse_tensor_dense_conv_3d.so",

    srcs =  [
              "sparse_tensor_dense_conv_3d.cc",
              "sparse_tensor_sparse_kernel_dense_conv_3d.h",
              "sparse_tensor_sparse_kernel_dense_conv_3d.cc",
              "sparse_tensor_dense_relu_op.cc",
              "sparse_tensor_dense_relu_op_functor.h",
              "sparse_tensor_dense_relu_op_gpu.cu.cc",
              "sparse_tensor_dense_relu_op.h"
            ],


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


