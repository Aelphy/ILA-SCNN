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
//  .Attr("T: {float, double, int32, complex64, complex128}")
REGISTER_OP("SparseTensorSparseKernelDenseConvKDFilterGrad")
  .Attr("T: {float}")
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
    Tensor gradients_ind = *gradients_indices; //non const working copy of gradient indices
    auto grad_ind = gradients_ind.matrix<int64>();
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

    size_t idx = 0;
    for(auto it = backprops.begin(); it != backprops.end(); ++it, ++idx){
      out_vals(idx) = *it;
      //TODO: assert input indice == out_vals first
    }
  }

 private:
  std::vector<int32> stride_;
  int32 filter_dim;
  std::string padding;
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConvKDFilterGrad").Device(DEVICE_CPU), \
  SparseTensorSparseKernelDenseConvKDFilterGrad<CPUDevice, type>);

REGISTER_CPU(float);
//REGISTER_CPU(double);
//REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);

