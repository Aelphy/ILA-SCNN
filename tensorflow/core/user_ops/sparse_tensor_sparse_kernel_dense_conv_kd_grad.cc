#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_sparse_kernel_dense_conv_kd_grad.h"


/** SparseTensorSparseKernelDenseConvKDFilterGrad
  * \ingroup CXX11_NeuralNetworks_Module
  * 
  * \brief Applies gradients of a 3D convolution over a sparse multichannel input voxel block.
  * 
  * The input parameter is expected to be a tensor with a rank of 4 or more (channels, depth, height, width, and optionally others).
  * The kernel parameter is expected to be a 5D tensor (filters, channels, kernel_depth, kernel_height, kernel_width).
  * 
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be filters, depth, height, width (and others if applicable).
  */


//TODO: How do I use REGISTER_OP with parameter T?
REGISTER_OP("SparseTensorSparseKernelDenseConvKDFilterGrad")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Input("gradients_indices: int64")
  .Input("gradients: T")
  .Input("gradients_shape: int64")
  .Output("backprops: T")
  .Attr("strides: list(int)")
  .Attr("padding: string = 'SAME'")
  .Attr("filter_dim: int = 3");


REGISTER_OP("SparseTensorSparseKernelDenseConvKDInputGrad")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Input("gradients_indices: int64")
  .Input("gradients: T")
  .Input("gradients_shape: int64")
  .Output("backprops: T")
  .Attr("strides: list(int)")
  .Attr("padding: string = 'SAME'")
  .Attr("filter_dim: int = 3");


REGISTER_OP("SparseTensorSparseKernelDenseConvKDGradV2")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Input("gradients_indices: int64")
  .Input("gradients: T")
  .Input("gradients_shape: int64")
  .Output("backprops_input: T")
  .Output("backprops_filter: T")
  .Attr("strides: list(int)")
  .Attr("padding: string = 'SAME'")
  .Attr("filter_dim: int = 3");

#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T>
class SparseTensorSparseKernelDenseConvKDFilterGrad : public OpKernel {
 public:
  explicit SparseTensorSparseKernelDenseConvKDFilterGrad(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_dim", &filter_dim));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape, *gradients, *gradients_indices, *gradients_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
    OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
    OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    OP_REQUIRES_OK(context, context->input("gradients_indices", &gradients_indices));
    OP_REQUIRES_OK(context, context->input("gradients_shape", &gradients_shape));
    auto in_ind = in_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    auto f_ind = filter_indices->matrix<int64>(); //filters, channels, kernel_depth, kernel_height, kernel_width TODO: other cases?
    auto f_vals = filter_values->flat<T>();
    auto f_sh = filter_shape->flat<int64>();
    Tensor grad_indices = *gradients_indices;
    auto grad_ind = grad_indices.matrix<int64>();
    auto grads = gradients->flat<T>();
    auto grad_sh = gradients_shape->flat<int64>();

    // Create an output tensor

    std::vector<T> backprops; //stores the values for the output tensor
    std::vector<int64> out_shape;
    sparseCuboidConvKDFilterGrad(in_ind, in_vals, in_sh, f_ind, f_vals, f_sh, grad_ind, grads, grad_sh, stride_, padding, filter_dim, backprops);
    
    Tensor *sparse_values = NULL;
    TensorShape out_val_shape = {(int64) f_vals.size()};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &sparse_values));
    auto out_vals = sparse_values->flat<T>();

    assert(backprops.size() == out_vals.size());

    for(size_t idx = 0; idx < backprops.size(); ++idx){
      out_vals(idx) = backprops[idx];
    }
  }

 private:
  std::vector<int32> stride_;
  int32 filter_dim;
  std::string padding;
};

template <typename Device, typename T>
class SparseTensorSparseKernelDenseConvKDInputGrad : public OpKernel {
 public:
  explicit SparseTensorSparseKernelDenseConvKDInputGrad(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_dim", &filter_dim));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape, *gradients, *gradients_indices, *gradients_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
    OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
    OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    OP_REQUIRES_OK(context, context->input("gradients_indices", &gradients_indices));
    OP_REQUIRES_OK(context, context->input("gradients_shape", &gradients_shape));
    auto in_ind = in_indices->matrix<int64>(); //batch, depth, height, width, channels, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    Tensor filter_ind = *filter_indices;
    auto f_ind = filter_ind.matrix<int64>(); //kernel_depth, kernel_height, kernel_width, in_channels, out_channels TODO: other cases?
    auto f_vals = filter_values->flat<T>();
    auto f_sh = filter_shape->flat<int64>();
    Tensor grad_indices = *gradients_indices;
    auto grad_ind = grad_indices.matrix<int64>();
    auto grads = gradients->flat<T>();
    auto grad_sh = gradients_shape->flat<int64>();

    // Create an output tensor

    std::vector<T> backprops; //stores the values for the output tensor
    std::vector<int64> out_shape;
    sparseCuboidConvKDInputGrad(in_ind, in_vals, in_sh, f_ind, f_vals, f_sh, grad_ind, grads, grad_sh, stride_, padding, filter_dim, backprops);
    
    Tensor *sparse_values = NULL;
    TensorShape out_val_shape = {(int64) in_vals.size()};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &sparse_values));
    auto out_vals = sparse_values->flat<T>();

    assert(backprops.size() == out_vals.size());

    for(size_t idx = 0; idx < backprops.size(); ++idx){
      out_vals(idx) = backprops[idx];
    }
  }

 private:
  std::vector<int32> stride_;
  int32 filter_dim;
  std::string padding;
};


template <typename Device, typename T>
class SparseTensorSparseKernelDenseConvKDGradV2 : public OpKernel {
 public:
  explicit SparseTensorSparseKernelDenseConvKDGradV2(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_dim", &filter_dim));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape, *gradients, *gradients_indices, *gradients_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
    OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
    OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
    OP_REQUIRES_OK(context, context->input("gradients", &gradients));
    OP_REQUIRES_OK(context, context->input("gradients_indices", &gradients_indices));
    OP_REQUIRES_OK(context, context->input("gradients_shape", &gradients_shape));
    auto in_ind = in_indices->matrix<int64>(); //batch, depth, height, width, channels, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    auto f_ind = filter_indices->matrix<int64>(); //kernel_depth, kernel_height, kernel_width, in_channels, out_channels TODO: other cases?
    auto f_vals = filter_values->flat<T>();
    auto f_sh = filter_shape->flat<int64>();
    auto grad_ind = gradients_indices->matrix<int64>();
    auto grads = gradients->flat<T>();
    auto grad_sh = gradients_shape->flat<int64>();


    // Create an output tensor
    Tensor *sparse_bp_fl = NULL, *sparse_bp_in = NULL;
    TensorShape out_in_shape = {(int64) in_vals.size()};
    TensorShape out_fl_shape = {(int64) f_vals.size()};
    OP_REQUIRES_OK(context, context->allocate_output("backprops_input", out_in_shape, &sparse_bp_in));
    OP_REQUIRES_OK(context, context->allocate_output("backprops_filter", out_fl_shape, &sparse_bp_fl));
    auto out_in_vals = sparse_bp_in->flat<T>();
    auto out_fl_vals = sparse_bp_fl->flat<T>();

    sparseCuboidConvKDBackpropV2<decltype(in_ind),decltype(in_vals),decltype(in_sh),T,decltype(out_fl_vals)>(in_ind, in_vals, in_sh, f_ind, f_vals, f_sh, grad_ind, grads, grad_sh, stride_, filter_dim, out_fl_vals,out_in_vals, padding);
    
  }

 private:
  std::vector<int32> stride_;
  int32 filter_dim;
  std::string padding;
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConvKDFilterGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
  SparseTensorSparseKernelDenseConvKDFilterGrad<CPUDevice, type>); \
  \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConvKDInputGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
  SparseTensorSparseKernelDenseConvKDInputGrad<CPUDevice, type>); \
  \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConvKDGradV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
  SparseTensorSparseKernelDenseConvKDGradV2<CPUDevice, type>); \

REGISTER_CPU(float);
//REGISTER_CPU(double);
//REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);
#undef REGISTER_CPU
