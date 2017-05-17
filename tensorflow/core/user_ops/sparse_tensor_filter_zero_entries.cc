#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_sparse_kernel_dense_conv_kd.h"


REGISTER_OP("SparseFilterZeroOp")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Output("out_indices: int64")
  .Output("out_values: T")
  .Output("out_shape: int64");


REGISTER_OP("SparseFilterZeroGradOp")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("out_indices: int64")
  .Input("gradients: T")
  .Output("backprops: T");


#include "tensorflow/core/framework/op_kernel.h"



using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class SparseFilterZeroOp : public OpKernel {
 public:
  explicit SparseFilterZeroOp(OpKernelConstruction* context) : OpKernel(context) 
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

    int64 non_zero_cnt = 0;
    //filter zeros in output
    for(size_t i = 0; i <in_vals.dimension(0); ++i){
      if(in_vals(i) > 0) non_zero_cnt++;
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
    for(size_t i = 0; i <in_vals.dimension(0); ++i){
      if(in_vals(i) > 0){
        out_vals(idx) = in_vals(i);
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

template <typename Device, typename T>
class SparseFilterZeroGradOp : public OpKernel {
 public:
  explicit SparseFilterZeroGradOp(OpKernelConstruction* context) : OpKernel(context) 
  {}

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *gradients, *out_indices;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    auto in_ind = in_indices->matrix<int64>();
    auto out_grads = gradients->flat<T>();
    auto out_ind = out_indices->matrix<int64>();
    assert(out_ind.dimension(0) <= in_ind.dimension(0));

    // Create an output tensor
    Tensor *sparse_values = NULL;
    TensorShape out_val_shape = {(int64) in_ind.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &sparse_values));
    auto out_vals = sparse_values->flat<T>();

    //find matches between input and output: reverse effects of filtering of 0 entries in SparseReluOp
    for(int64 i = 0; i < out_vals.dimension(0); ++i){
        out_vals(i) = 0; //TODO: not nice... use allocator
    }
    for(int64 i = 0, j = 0; i < in_ind.dimension(0) && j < out_ind.dimension(0); ++i){
      bool match = true;
      for(int64 k = 0; k < in_ind.dimension(1); ++k){
        match &= (in_ind(i,k) == out_ind(j,k));
      }
      if(match){
        out_vals(i) = out_grads(j);
        j++;
      }
    }
  }
};

#define REGISTER_CPU(type)                                                                             \
  REGISTER_KERNEL_BUILDER(Name("SparseFilterZeroOp").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
                            SparseFilterZeroOp<CPUDevice, type>);                                      \ 
  REGISTER_KERNEL_BUILDER(Name("SparseFilterZeroGradOp").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
                            SparseFilterZeroGradOp<CPUDevice, type>);

  REGISTER_CPU(float);
  REGISTER_CPU(double);
  REGISTER_CPU(int32);
  //REGISTER_CPU(complex64);
  //REGISTER_CPU(complex128);
#undef REGISTER_CPU

