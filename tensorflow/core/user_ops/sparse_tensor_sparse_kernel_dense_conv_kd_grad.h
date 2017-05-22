#pragma once

#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "sparse_tensor_sparse_kernel_dense_conv_kd.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename GIndexT, typename GradT, typename T, typename IndiceT> void
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
                        const IndiceT dim,
                        std::vector<T>& back_props) 
  {

    //sparse back propagation (only backpropergate errors to features or filter weights, which exist (unequal zero)):
    typedef std::vector<IndiceT> KeyT;

    // 1. reverse strides and effects of padding to make grad (output layer) and in_vals (input layer) comparable
    KeyT filter_offset(f_sh.dimension(0), 0);
    for(size_t i = 1; i < dim + 1; ++i){
      filter_offset[i] = (f_sh(i - 1) - 1) / 2;
    }
    KeyT in_shape(in_sh.dimension(0), 0);
    for(size_t i = 0; i < in_shape.size(); ++i){
      in_shape[i] = in_sh(i);
    }

    //reverse effects of tensorflows (SAME) padding rule
    std::vector<int> str_padding_offset(in_ind.dimension(1), 0);
    for(IndiceT i = 0; i < str_padding_offset.size(); ++i){
      if(int(in_sh(i)) % stride_[i] == 1){
        str_padding_offset[i] = 0;
      }
    }


    GIndexT adapted_grads_ind = grads_ind;
    auto adapted_grads_ptr = &adapted_grads_ind; auto stride_ptr = &stride_; auto padding_ptr = &padding;
#pragma omp parallel for firstprivate(adapted_grads_ptr, filter_offset, str_padding_offset, dim, padding_ptr, stride_ptr)
    for(size_t i = 0; i < (*adapted_grads_ptr).dimension(0); ++i){
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, output_channels, in_channels]
      for(size_t j = 1; j < dim + 1; ++j){
        (*adapted_grads_ptr)(i,j) = (*adapted_grads_ptr)(i,j) * (*stride_ptr)[j]; //reverse stride on indices
        if((*padding_ptr) == "VALID"){
          (*adapted_grads_ptr)(i,j) += filter_offset[j]; //reverse VALID padding on indices
        } else if((*stride_ptr)[j] > 1) {
          (*adapted_grads_ptr)(i,j) += str_padding_offset[j]; //reverse SAME padding rules from tensorflow
        }
      }
    }
    

    // 2. set up convolution for backprop of filter weights: in_vals * grads
    KeyT grad_offset(f_sh.dimension(0),0);
    for(size_t i = 1; i < dim + 1; ++i){
      grad_offset[i] = (grads_sh(i) - 1) / 2;
    }

    std::vector<int32> no_stride(stride_.size(), 1);
    ConvKDHelper<IndexT, GIndexT, ValueT, KeyT, T, IndiceT> conv_function(  &in_ind, 
                                                                  &in_vals, 
                                                                  &adapted_grads_ind, 
                                                                  &grads,
                                                                  //in_shape,
                                                                  &no_stride, 
                                                                  "SAME",
                                                                  grad_offset,
                                                                  dim);

    // 3. convolve grads and input values corresponding to filter index
    back_props.resize(f_ind.dimension(0));
    auto back_props_ptr = &back_props; auto f_ind_ptr = &f_ind; auto conv_ptr = &conv_function;
#pragma omp parallel for firstprivate(back_props_ptr, f_ind_ptr, conv_ptr, grad_offset, filter_offset, dim)
    for(size_t i = 0; i < f_ind_ptr->dimension(0); ++i){
      KeyT id(f_ind_ptr->dimension(1), 0);
      for(size_t j = 1; j < dim + 1; ++j){
        id[j - 1] = (*f_ind_ptr)(i,j-1) + grad_offset[j] - filter_offset[j];
      }
      id[dim + 1] = (*f_ind_ptr)(i, dim + 1); //in channels;
      id[dim] = (*f_ind_ptr)(i, dim); //out channels;
      (*back_props_ptr)[i] = conv_ptr->backprop_filter_at(id); //normal convolution (no voting sheme) for large matrices with VALID padding + zeros
    }
  }

  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename GIndexT, typename GradT, typename T, typename IndiceT> void
  sparseCuboidConvKDInputGrad(   const IndexT& in_ind, 
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
                        const IndiceT dim,
                        std::vector<T>& back_props) 
  {
    //sparse back propagation (only backpropergate errors to features or filter weights, which exist (unequal zero)):
    typedef std::vector<IndiceT> KeyT;

    // 1. reverse strides and effects of padding to make grad (output layer) and in_vals (input layer) comparable
    KeyT filter_offset(f_sh.dimension(0), 0);
    for(size_t i = 1; i < dim + 1; ++i){
      filter_offset[i] = (f_sh(i - 1) - 1) / 2;
    }
    KeyT grads_shape(grads_sh.dimension(0), 0);
    for(size_t i = 0; i < grads_shape.size(); ++i){
      grads_shape[i] = grads_sh(i);
    }
    //reverse effects of tensorflows (SAME) padding rule on grads
    std::vector<int> str_padding_offset(in_ind.dimension(1), 0);
    for(IndiceT i = 0; i < str_padding_offset.size(); ++i){
      if(int(in_sh(i)) % stride_[i] == 1){
        str_padding_offset[i] = 0;
      }
    }

    GIndexT adapted_grads_ind = grads_ind;
    auto adapted_grads_ptr = &adapted_grads_ind; auto stride_ptr = &stride_; auto padding_ptr = &padding;
#pragma omp parallel for firstprivate(adapted_grads_ptr, dim, filter_offset, str_padding_offset, padding_ptr, stride_ptr)
    for(size_t i = 0; i < (*adapted_grads_ptr).dimension(0); ++i){
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, output_channels, in_channels]
      for(size_t j = 1; j < dim + 1; ++j){
        (*adapted_grads_ptr)(i,j) = (*adapted_grads_ptr)(i,j) * (*stride_ptr)[j]; //reverse stride on indices
        if((*padding_ptr) == "VALID"){
          (*adapted_grads_ptr)(i,j) += filter_offset[j]; //reverse VALID padding on indices
        } else if((*stride_ptr)[j] > 1) {
          (*adapted_grads_ptr)(i,j) += str_padding_offset[j]; //reverse SAME padding rules from tensorflow
        }
      }
    }

    //transpose filter indices and convert into input format
    GIndexT adapted_f_ind = f_ind;
    for(size_t i = 0; i < adapted_f_ind.dimension(0); ++i){
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, output_channels, in_channels]
      for(size_t j = 0; j < dim; ++j){
        adapted_f_ind(i,j) = f_sh(j) - 1 - adapted_f_ind(i,j); //transpose of filter
      }
    }

    // 2. set up convolution for backprop of filter weights: in_vals * grads
    KeyT grad_offset(f_sh.dimension(0),0);
    for(size_t i = 1; i < dim + 1; ++i){
      grad_offset[i] = (grads_sh(i) - 1) / 2;
    }

    std::vector<int32> no_stride(stride_.size(), 1);
    ConvKDHelper<FIndexT, GIndexT, ValueT, KeyT, T, IndiceT> conv_function(&adapted_grads_ind, 
                                                                  &grads, 
                                                                  &adapted_f_ind, 
                                                                  &f_vals,
                                                                  //grads_shape,
                                                                  &no_stride, 
                                                                  "SAME",
                                                                  filter_offset,
                                                                  dim);

    // 3. convolve grads and input values corresponding to filter index
    back_props.resize(in_ind.dimension(0));
    auto back_props_ptr = &back_props; auto in_ind_ptr = &in_ind; auto conv_ptr = &conv_function;
#pragma omp parallel for firstprivate(back_props_ptr, in_ind_ptr, conv_ptr, dim)
    for(size_t i = 0; i < in_ind_ptr->dimension(0); ++i){
      KeyT id(in_ind_ptr->dimension(1), 0);
      for(size_t j = 1; j < dim + 1; ++j){
        id[j - 1] = (*in_ind_ptr)(i,j);
      }
      id[dim + 1] = (*in_ind_ptr)(i, dim + 1); //in channels;
      id[dim] = (*in_ind_ptr)(i, 0); //batch;
      (*back_props_ptr)[i] = conv_ptr->backprop_indices_at(id); //normal convolution (no voting sheme) for large matrices with VALID padding + zeros
    }
  }
}
