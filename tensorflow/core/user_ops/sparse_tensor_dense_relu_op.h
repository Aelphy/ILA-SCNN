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

#pragma once

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "sparse_tensor_dense_relu_op_functor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

template <typename Device, typename T>
class SparseReluOp : public UnaryElementWiseOp<T, SparseReluOp<Device, T>> {
 public:
  using UnaryElementWiseOp<T, SparseReluOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::SparseRelu<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

// Out of line check to save code space (we have this code once, rather
// than once for every NDIMS * NumTypes * Num_different_relu_variants
// functions.
struct SparseReluHelpers {
  static void ValidateSameSizeHelper(OpKernelContext* context, const Tensor& g,
                                     const Tensor& a) {
    OP_REQUIRES(context, a.IsSameSize(g),
                errors::InvalidArgument("g and a must be the same size"));
  }
  static bool ValidateSameSize(OpKernelContext* context, const Tensor& g,
                               const Tensor& a) {
    ValidateSameSizeHelper(context, g, a);
    return context->status().ok();
  }
};

template <typename Device, typename T>
class SparseReluGradOp : public BinaryElementWiseOp<T, SparseReluGradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, SparseReluGradOp<Device, T>>::BinaryElementWiseOp;

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): either the inputs that were passed to SparseReluOp(), or its
  //               outputs (using either one yields the same result here).
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void SparseReluGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                              const Tensor& g, const Tensor& a,
                                              Tensor* output) {
  if (!SparseReluHelpers::ValidateSameSize(context, g, a)) return;
  functor::SparseReluGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

template <typename Device, typename T>
class SparseRelu6Op : public UnaryElementWiseOp<T, SparseRelu6Op<Device, T>> {
 public:
  using UnaryElementWiseOp<T, SparseRelu6Op<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::SparseRelu6<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class SparseRelu6GradOp : public BinaryElementWiseOp<T, SparseRelu6GradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, SparseRelu6GradOp<Device, T>>::BinaryElementWiseOp;

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to SparseRelu6Op()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void SparseRelu6GradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                               const Tensor& g, const Tensor& a,
                                               Tensor* output) {
  if (!SparseReluHelpers::ValidateSameSize(context, g, a)) return;
  functor::SparseRelu6Grad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

template <typename Device, typename T>
class SparseEluOp : public UnaryElementWiseOp<T, SparseEluOp<Device, T>> {
 public:
  using UnaryElementWiseOp<T, SparseEluOp<Device, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::SparseElu<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class SparseEluGradOp : public BinaryElementWiseOp<T, SparseEluGradOp<Device, T>> {
 public:
  using BinaryElementWiseOp<T, SparseEluGradOp<Device, T>>::BinaryElementWiseOp;

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (outputs): outputs of the SparseEluOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void SparseEluGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                             const Tensor& g, const Tensor& a,
                                             Tensor* output) {
  if (!SparseReluHelpers::ValidateSameSize(context, g, a)) return;
  functor::SparseEluGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

}  // namespace tensorflow

#undef EIGEN_USE_THREADS
