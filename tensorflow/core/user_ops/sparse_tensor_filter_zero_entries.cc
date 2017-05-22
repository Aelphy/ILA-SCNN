#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_sparse_kernel_dense_conv_kd.h"


REGISTER_OP("SparseFilterZeroOp")
  .Attr("Tindices: {int32, int64}")
  .Attr("T: realnumbertype")
  .Input("in_indices: Tindices")
  .Input("in_values: T")
  .Input("in_shape: Tindices")
  .Output("out_indices: Tindices")
  .Output("out_values: T")
  .Output("out_shape: Tindices");


REGISTER_OP("SparseFilterZeroGradOp")
  .Attr("Tindices: {int32, int64}")
  .Attr("T: realnumbertype")
  .Input("in_indices: Tindices")
  .Input("out_indices: Tindices")
  .Input("gradients: T")
  .Output("backprops: T");


#include "tensorflow/core/framework/op_kernel.h"



using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, typename Tindices>
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
    auto in_ind = in_indices->matrix<Tindices>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<Tindices>();

    Tindices non_zero_cnt = 0;
    //filter zeros in output
    for(size_t i = 0; i <in_vals.dimension(0); ++i){
      if(in_vals(i) > 0) non_zero_cnt++;
    }

    // Create an output tensor
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
    TensorShape out_ind_shape = {non_zero_cnt, (Tindices) in_ind.dimension(1)};
    TensorShape out_val_shape = {non_zero_cnt};
    TensorShape out_sh_shape = {(Tindices) in_ind.dimension(1)};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &sparse_shape));

    auto out_ind = sparse_indices->matrix<Tindices>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto out_vals = sparse_values->flat<T>();
    auto out_sh = sparse_shape->flat<Tindices>();

    //filter zeros
    Tindices idx = 0;
    for(size_t i = 0; i <in_vals.dimension(0); ++i){
      if(in_vals(i) > 0){
        out_vals(idx) = in_vals(i);
        for(size_t j = 0; j < in_ind.dimension(1); ++j){
          out_ind(idx,j) = in_ind(i,j);
        }
        idx++;
      }
    }
    for(Tindices idx = 0; idx < in_ind.dimension(1); ++idx){
        out_sh(idx) = in_sh(idx);
    }
  }
};

template <typename Device, typename T, typename Tindices>
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
    auto in_ind = in_indices->matrix<Tindices>();
    auto out_grads = gradients->flat<T>();
    auto out_ind = out_indices->matrix<Tindices>();
    assert(out_ind.dimension(0) <= in_ind.dimension(0));

    // Create an output tensor
    Tensor *sparse_values = NULL;
    TensorShape out_val_shape = {(Tindices) in_ind.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &sparse_values));
    auto out_vals = sparse_values->flat<T>();

    //find matches between input and output: reverse effects of filtering of 0 entries in SparseReluOp
    for(Tindices i = 0; i < out_vals.dimension(0); ++i){
        out_vals(i) = 0; //TODO: not nice... use allocator
    }
    for(Tindices i = 0, j = 0; i < in_ind.dimension(0) && j < out_ind.dimension(0); ++i){
      bool match = true;
      for(Tindices k = 0; k < in_ind.dimension(1); ++k){
        match &= (in_ind(i,k) == out_ind(j,k));
      }
      if(match){
        out_vals(i) = out_grads(j);
        j++;
      }
    }
  }
};

#define REGISTER_CPU_TYPE(type, indices_type)                                                               \
  REGISTER_KERNEL_BUILDER(Name("SparseFilterZeroOp").Device(DEVICE_CPU).TypeConstraint<type>("T").TypeConstraint<indices_type>("Tindices"),     \
                            SparseFilterZeroOp<CPUDevice, type, indices_type>);                                      \ 
  REGISTER_KERNEL_BUILDER(Name("SparseFilterZeroGradOp").Device(DEVICE_CPU).TypeConstraint<type>("T").TypeConstraint<indices_type>("Tindices"), \
                            SparseFilterZeroGradOp<CPUDevice, type, indices_type>);

#define REGISTER_CPU_ALL(type) \
  REGISTER_CPU_TYPE(type, int64); \
  REGISTER_CPU_TYPE(type, int32);

  REGISTER_CPU_ALL(float);
  REGISTER_CPU_ALL(double);
  REGISTER_CPU_ALL(int32);
  //REGISTER_CPU_ALL(complex64);
  //REGISTER_CPU_ALL(complex128);
#undef REGISTER_CPU_ALL
#undef REGISTER_CPU_TYPE

