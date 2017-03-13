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

REGISTER_OP("Relu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear: `max(features, 0)`.
)doc");

REGISTER_OP("ReluGrad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear gradients for a Relu operation.

gradients: The backpropagated gradients to the corresponding Relu operation.
features: The features passed as input to the corresponding Relu operation, OR
  the outputs of that operation (both work equivalently).
backprops: `gradients * (features > 0)`.
)doc");

REGISTER_OP("Relu6")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear 6: `min(max(features, 0), 6)`.
)doc");

REGISTER_OP("Relu6Grad")
    .Input("gradients: T")
    .Input("features: T")
    .Output("backprops: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes rectified linear 6 gradients for a Relu6 operation.

gradients: The backpropagated gradients to the corresponding Relu6 operation.
features: The features passed as input to the corresponding Relu6 operation.
backprops: The gradients:
  `gradients * (features > 0) * (features < 6)`.
)doc");

REGISTER_OP("Elu")
    .Input("features: T")
    .Output("activations: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
](http://arxiv.org/abs/1511.07289)
)doc");

REGISTER_OP("EluGrad")
    .Input("gradients: T")
    .Input("outputs: T")
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

#define REGISTER_RELU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      SparseReluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      SparseReluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseRelu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SparseRelu6Op<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseRelu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseRelu6GradOp<CPUDevice, type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);
#undef REGISTER_RELU_KERNELS

#define REGISTER_ELU_KERNELS(type)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SparseElu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SparseEluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("SparseEluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseEluGradOp<CPUDevice, type>)

// SparseElu only makes sense with float or double.
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ELU_KERNELS);
#undef REGISTER_ELU_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void SparseRelu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct SparseRelu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void SparseReluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct SparseReluGrad<GPUDevice, T>;                               \
                                                                               \
  template <>                                                                  \
  void SparseRelu6<GPUDevice, T>::operator()(                                        \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct SparseRelu6<GPUDevice, T>;                                  \
                                                                               \
  template <>                                                                  \
  void SparseRelu6Grad<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct SparseRelu6Grad<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void SparseElu<GPUDevice, T>::operator()(const GPUDevice& d,                       \
                                     typename TTypes<T>::ConstTensor features, \
                                     typename TTypes<T>::Tensor activations);  \
  extern template struct SparseElu<GPUDevice, T>;                                    \
                                                                               \
  template <>                                                                  \
  void SparseEluGrad<GPUDevice, T>::operator()(                                      \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct SparseEluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseRelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      SparseReluOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
      SparseReluGradOp<GPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseRelu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      SparseRelu6Op<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseRelu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SparseRelu6GradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseElu").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      SparseEluOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("SparseEluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),   \
      SparseEluGradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
