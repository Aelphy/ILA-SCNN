/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "sparse_tensor_dense_relu_op_functor.h"


REGISTER_OP("SparseRelu")
    .Input("in_indices: int64")
    .Input("in_values: T")
    .Input("in_shape: int64")
  	.Output("sparse_indices: int64")
  	.Output("sparse_values: T")
  	.Output("sparse_shape: int64")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear: `max(features, 0)`.
)doc");

REGISTER_OP("SparseReluGrad")
    .Input("in_indices: int64")
    .Input("in_values: T")
  	.Input("out_indices: int64")
    .Input("gradients: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear gradients for a Relu operation.

gradients: The backpropagated gradients to the corresponding Relu operation.
features: The features passed as input to the corresponding Relu operation, OR
  the outputs of that operation (both work equivalently).
backprops: `gradients * (features > 0)`.
)doc");

REGISTER_OP("SparseRelu6")
    .Input("in_indices: int64")
    .Input("in_values: T")
    .Input("in_shape: int64")
  	.Output("sparse_indices: int64")
  	.Output("sparse_values: T")
  	.Output("sparse_shape: int64")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear 6: `min(max(features, 0), 6)`.
)doc");

REGISTER_OP("SparseRelu6Grad")
    .Input("in_indices: int64")
    .Input("in_values: T")
    .Input("out_indices: int64")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear 6 gradients for a Relu6 operation.

gradients: The backpropagated gradients to the corresponding Relu6 operation.
features: The features passed as input to the corresponding Relu6 operation.
backprops: The gradients:
  `gradients * (features > 0) * (features < 6)`.
)doc");

REGISTER_OP("SparseElu")
    .Input("in_indices: int64")
    .Input("in_values: T")
    .Input("in_shape: int64")
  	.Output("sparse_indices: int64")
  	.Output("sparse_values: T")
  	.Output("sparse_shape: int64")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)
)doc");

REGISTER_OP("SparseEluGrad")
    .Input("in_indices: int64")
    .Input("out_indices: int64")
    .Input("out_values: T")
    .Input("gradients: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes gradients for the exponential linear (Elu) operation.

gradients: The backpropagated gradients to the corresponding Elu operation.
outputs: The outputs of the corresponding Elu operation.
backprops: The gradients: `gradients * (outputs + 1)` if outputs < 0,
`gradients` otherwise.
)doc");

#include "sparse_tensor_dense_relu_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_RELU_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      SparseReluOp<CPUDevice, type, functor::Relu>);                        \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseRelu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SparseReluOp<CPUDevice, type, functor::Relu6>);                       \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseElu").Device(DEVICE_CPU).TypeConstraint<type>("T"),       \
      SparseReluOp<CPUDevice, type, functor::Elu>);                         \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      SparseReluGradOp<CPUDevice, type, functor::ReluGrad>);                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseRelu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseReluGradOp<CPUDevice, type, functor::Relu6Grad>)                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseEluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      SparseEluGradOp<CPUDevice, type, functor::EluGrad>)


REGISTER_RELU_KERNELS(float)
REGISTER_RELU_KERNELS(double)
REGISTER_RELU_KERNELS(int64)
REGISTER_RELU_KERNELS(int)
//TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);

}  // namespace tensorflow
