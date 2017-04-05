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
    //sparse back propagation (only backpropergate errors to features or filter weights, which exist (unequal zero)):
    typedef std::vector<int64> KeyT;
    typedef std::vector<KeyT> GIndexKeyT;

    // 1. reverse strides and effects of padding to make grad (output layer) and in_vals (input layer) comparable
    KeyT filter_offset(f_sh.dimension(0), 0);
    for(size_t i = 1; i < dim + 1; ++i){
      filter_offset[i] = (f_sh(i - 1) - 1) / 2;
    }

    //reverse effects of tensorflows (SAME) padding rule
    std::vector<int> str_padding_offset(in_ind.dimension(1), 0);
    for(int64 i = 0; i < str_padding_offset.size(); ++i){
      if(int(in_sh(i)) % stride_[i] == 1){
        str_padding_offset[i] = 1;
      }
    }


    GIndexKeyT adapted_grads_ind(grads_ind.dimension(0), KeyT(grads_ind.dimension(1), 0));
    for(size_t i = 0; i < adapted_grads_ind.size(); ++i){
      for(size_t j = 0; j < grads_ind.dimension(1); ++j){
        adapted_grads_ind[i][j] = grads_ind(i,j);
      }
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, output_channels, in_channels]
      for(size_t j = 1; j < dim + 1; ++j){
        adapted_grads_ind[i][j] = adapted_grads_ind[i][j] * stride_[j]; //reverse stride on indices
        if(padding == "VALID"){
          adapted_grads_ind[i][j] += filter_offset[j]; //reverse VALID padding on indices
        } else if(stride_[j] > 1) {
          adapted_grads_ind[i][j] += str_padding_offset[j]; //reverse SAME padding rules from tensorflow
        }
      }
      //bring gradient indices from input format [batch, depth, height, width, in_channels] to filter format [depth, height, width, output_channels, in_channels]
      for(size_t j = 1; j < dim + 1; ++j){
        adapted_grads_ind[i][j-1] = adapted_grads_ind[i][j];
      }

      adapted_grads_ind[i][dim] = adapted_grads_ind[i][dim + 1]; //output_channels = in_channels
    }
    

    // 2. set up convolution for backprop of filter weights: in_vals * grads
    KeyT grad_offset(f_sh.dimension(0),0);
    for(size_t i = 1; i < dim + 1; ++i){
      grad_offset[i - 1] = (grads_sh(i) - 1) / 2;
    }

    std::vector<int32> no_stride(stride_.size(), 1);
    ConvKDHelper<IndexT, GIndexKeyT, ValueT, KeyT, T> conv_function(  &in_ind, 
                                                                  &in_vals, 
                                                                  &adapted_grads_ind, 
                                                                  &grads, 
                                                                  &no_stride, 
                                                                  "SAME",
                                                                  grad_offset,
                                                                  dim);

    // 3. convolve grads and input values corresponding to filter index
    back_props.resize(f_ind.dimension(0));
    for(size_t i = 0; i < f_ind.dimension(0); ++i){
      KeyT id(f_ind.dimension(1), 0);
      for(size_t j = 1; j < dim + 1; ++j){
        id[j] = f_ind(i,j-1) + grad_offset[j - 1] - filter_offset[j];
      }
      back_props[i] = conv_function.evaluate_at(id); //normal convolution (no voting sheme) for large matrices with VALID padding + zeros
    }
  }
}
