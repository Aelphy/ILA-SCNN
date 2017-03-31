#pragma once

#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "sparse_tensor_sparse_kernel_dense_conv_kd.h"

namespace tensorflow {
  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename GIndexT, typename GradT, typename T> void
  sparseCuboidConvKDFilterGrad(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const FIndexT& f_ind, 
                        const FValueT& f_vals, 
                        const FShapeT& f_sh,
                        const GIndexT& grads_ind,
                        const GradT& grads,
                        const ShapeT& grads_sh,
                        const std::vector<int32>& stride_,
                        const string& padding,
                        const int64 dim,
                        std::vector<T>& back_props) 
  {
    //zero-truncated back propagation (only backpropergate errors to features or filter weights, which exist (unequal zero)):
    typedef std::vector<int64> KeyT;

    // 1. reverse strides and effects of VALID padding to make grad (output layer) and in_vals (input layer) comparable
    KeyT filter_offset(f_sh.dimension(0),0);
    for(size_t i = 1; i < dim + 1; ++i){
      filter_offset[i] = (f_sh(i - 1) - 1) / 2;
    }
    auto adapted_grads_ind = grads_ind; //const index
    for(size_t i = 0; i < adapted_grads_ind.dimension(0); ++i){
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, output_channels, in_channels]
      for(size_t j = 1; j < dim + 1; ++j){
        adapted_grads_ind(i,j) = adapted_grads_ind(i,j) * stride_[j]; //reverse stride on indices
        if(padding == "VALID"){
          adapted_grads_ind(i,j) += filter_offset[j]; //reverse VALID padding on indices
        }
      }
    }
    

    // 2. set up convolution for backprop of filter weights: in_vals * grads
    ConvKDHelper<IndexT, GIndexT, ValueT, ShapeT, KeyT, T> conv_function(  &in_ind, 
                                                                  &in_vals, 
                                                                  &in_sh, 
                                                                  &adapted_grads_ind, 
                                                                  &grads, 
                                                                  &grads_sh, 
                                                                  &stride_, 
                                                                  "SAME", 
                                                                  dim);
    // 3. convolve grads and input values corresponding to filter index
    KeyT grad_offset(f_sh.dimension(0),0);
    for(size_t i = 1; i < dim + 1; ++i){
      grad_offset[i] = (grads_sh(i - 1) - 1) / 2;
    }
    back_props.resize(f_ind.dimension(0));
    for(size_t i = 0; i < f_ind.dimension(0); ++i){
      KeyT id(f_ind.dimension(1), 0);
      for(size_t j = 0; j < id.size(); ++j){
        id[j] = f_ind(i,j) - filter_offset[j] + grad_offset[j];
      }
      back_props[i] = conv_function.evaluate_at(id);
    }
  }
}
