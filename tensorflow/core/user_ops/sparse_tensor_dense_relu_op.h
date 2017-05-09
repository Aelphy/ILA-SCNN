#pragma once

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

template <typename Device, typename T, template <typename, typename> class FunctorT>
class SparseReluOp : public OpKernel {
 public:
  explicit SparseReluOp(OpKernelConstruction* context) : OpKernel(context) 
  {}

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    auto in_ind = in_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    auto device = context->eigen_device<Device>();

    Tensor tmp_out_values = *in_values;
    auto tmp_out_vals = tmp_out_values.flat<T>();

    //run Relu
    FunctorT<Device, T> fnct;
    fnct(device, in_vals, tmp_out_vals);
    int64 non_zero_cnt = 0;

    //filter zeros in output
    for(size_t i = 0; i <tmp_out_vals.dimension(0); ++i){
      if(tmp_out_vals(i) > 0) non_zero_cnt++;
    }

    // Create an output tensor
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
    TensorShape out_ind_shape = {non_zero_cnt, (int64) in_ind.dimension(1)};
    TensorShape out_val_shape = {non_zero_cnt};
    TensorShape out_sh_shape = {(int64) in_ind.dimension(1)};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &sparse_shape));

    auto out_ind = sparse_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto out_vals = sparse_values->flat<T>();
    auto out_sh = sparse_shape->flat<int64>();

    //filter zeros
    int64 idx = 0;
    for(size_t i = 0; i <tmp_out_vals.dimension(0); ++i){
      if(tmp_out_vals(i) > 0){
        out_vals(idx) = tmp_out_vals(i);
        for(size_t j = 0; j < in_ind.dimension(1); ++j){
          out_ind(idx,j) = in_ind(i,j);
        }
        idx++;
      }
    }
    for(int64 idx = 0; idx < in_ind.dimension(1); ++idx){
        out_sh(idx) = in_sh(idx);
    }
  }
};

template <typename Device, typename T, template <typename, typename> class FunctorT>
class SparseReluGradOp : public OpKernel {
 public:
  explicit SparseReluGradOp(OpKernelConstruction* context) : OpKernel(context) 
  {}

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *gradients, *out_indices;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    auto in_ind = in_indices->matrix<int64>();
    auto in_vals = in_values->flat<T>();
    auto out_grads = gradients->flat<T>();
    Tensor in_gradients = *in_values;
    auto in_grads = in_gradients.flat<T>();
    auto out_ind = out_indices->matrix<int64>();
    assert(out_ind.dimension(0) <= in_ind.dimension(0));

    //find matches between input and output: reverse effects of filtering of 0 entries in SparseReluOp
    for(int64 i = 0; i < in_grads.dimension(0); ++i){
        in_grads(i) = 0; //TODO: not nice... use allocator
    }
    for(int64 i = 0, j = 0; i < in_ind.dimension(0) && j < out_ind.dimension(0); ++i){
      bool match = true;
      for(int64 k = 0; k < in_ind.dimension(1); ++k){
        match &= (in_ind(i,k) == out_ind(j,k));
      }
      if(match){
        in_grads(i) = out_grads(j);
        j++;
      }
    }

    // Create an output tensor
    Tensor *sparse_values = NULL;
    TensorShape out_val_shape = {(int64) in_ind.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &sparse_values));
    auto out_vals = sparse_values->flat<T>();
  
    // back propergation
    auto device = context->eigen_device<Device>();
    const Tensor* cin_gradients = &in_gradients;
    auto cin_grads = cin_gradients->flat<T>();
    FunctorT<Device, T> fnct;
    fnct(device, cin_grads, in_vals, out_vals);
  }
};


template <typename Device, typename T, template <typename, typename> class FunctorT>
class SparseEluGradOp : public OpKernel {
 public:
  explicit SparseEluGradOp(OpKernelConstruction* context) : OpKernel(context) 
  {}

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *out_values, *gradients, *out_indices;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
    OP_REQUIRES_OK(context, context->input("out_values", &out_values));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    auto in_ind = in_indices->matrix<int64>();
    auto out_vals = out_values->flat<T>();
    auto out_grads = gradients->flat<T>();
    auto out_ind = out_indices->matrix<int64>();
    assert(out_ind.dimension(0) <= in_ind.dimension(0));
  
    // sparse back propergation
    auto device = context->eigen_device<Device>();
    Tensor tmp_out_values = *out_values;
    auto tmp_out_vals = tmp_out_values.flat<T>();
    FunctorT<Device, T> fnct;
    fnct(device, out_grads, out_vals, tmp_out_vals);

    // Create an output tensor
    Tensor *sparse_values = NULL;
    TensorShape out_val_shape = {(int64) in_ind.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &sparse_values));
    auto backprops = sparse_values->flat<T>();

    //find matches between input and output: reverse effects of filtering of 0 entries in SparseReluOp
    for(int64 i = 0; i < out_vals.dimension(0); ++i){
        backprops(i) = 0; //TODO: not nice... use allocator
    }
    for(int64 i = 0, j = 0; i < in_ind.dimension(0) && j < out_ind.dimension(0); ++i){
      bool match = true;
      for(int64 k = 0; k < in_ind.dimension(1); ++k){
        match &= (in_ind(i,k) == out_ind(j,k));
      }
      if(match){
        backprops(i) = tmp_out_vals(j);
        j++;
      }
    }
  }
};


}  // namespace tensorflow

#undef EIGEN_USE_THREADS
