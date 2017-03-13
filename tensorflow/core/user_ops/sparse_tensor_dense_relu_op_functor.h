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

#pragma once

// Functor definition for SparseReluOp and SparseReluGradOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by SparseReluOp to do the computations.
template <typename Device, typename T>
struct SparseRelu {
  // Computes SparseRelu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) = features.cwiseMax(static_cast<T>(0));
  }
};

// Functor used by SparseReluGradOp to do the computations.
template <typename Device, typename T>
struct SparseReluGrad {
  // Computes SparseReluGrad backprops.
  //
  // gradients: gradients backpropagated to the SparseRelu op.
  // features: either the inputs that were passed to the SparseRelu or, or its
  //           outputs (using either one yields the same result here).
  // backprops: gradients to backpropagate to the SparseRelu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    // NOTE: When the activation is exactly zero, we do not propagate the
    // associated gradient value. This allows the output of the SparseRelu to be used,
    // as well as its input.
    backprops.device(d) =
        gradients * (features > static_cast<T>(0)).template cast<T>();
  }
};

// Functor used by SparseRelu6Op to do the computations.
template <typename Device, typename T>
struct SparseRelu6 {
  // Computes SparseRelu6 activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) =
        features.cwiseMax(static_cast<T>(0)).cwiseMin(static_cast<T>(6));
  }
};

// Functor used by SparseReluGradOp to do the computations.
template <typename Device, typename T>
struct SparseRelu6Grad {
  // Computes SparseRelu6Grad backprops.
  //
  // gradients: gradients backpropagated to the SparseRelu6 op.
  // features: inputs that where passed to the SparseRelu6 op.
  // backprops: gradients to backpropagate to the SparseRelu6 inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    // NOTE: When the activation is exactly zero or six, we
    // arbitrarily choose to not propagate the associated gradient
    // value.
    backprops.device(d) =
        gradients *
        ((features > static_cast<T>(0)) * (features < static_cast<T>(6)))
            .template cast<T>();
  }
};

// Functor used by SparseEluOp to do the computations.
template <typename Device, typename T>
struct SparseElu {
  // Computes SparseElu activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    // features.constant(?)
    activations.device(d) =
        (features < static_cast<T>(0))
            .select(features.exp() - features.constant(static_cast<T>(1)),
                    features);
  }
};

// Functor used by SparseEluGradOp to do the computations.
template <typename Device, typename T>
struct SparseEluGrad {
  // Computes SparseEluGrad backprops.
  //
  // gradients: gradients backpropagated to the SparseElu op.
  // activations: outputs of the SparseElu op.
  // backprops: gradients to backpropagate to the SparseElu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor activations,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        (activations < static_cast<T>(0))
            .select((activations + static_cast<T>(1)) * gradients, gradients);
  }
};

}  // namespace functor
}  // namespace tensorflow

