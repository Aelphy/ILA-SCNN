#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_sparse_kernel_dense_conv_kd.h"


/** SparseTensorSparseKernelDenseConv3D
  * \ingroup CXX11_NeuralNetworks_Module
  * 
  * \brief Applies a 3D convolution over a multichannel input voxel block.
  * 
  * The input parameter is expected to be a tensor with a rank of 4 or more (channels, depth, height, width, and optionally others).
  * The kernel parameter is expected to be a 5D tensor (filters, channels, kernel_depth, kernel_height, kernel_width).
  * 
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be filters, depth, height, width (and others if applicable).
  */



REGISTER_OP("SparseTensorSparseKernelDenseConvKD")
  .Attr("T: realnumbertype")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Output("out_indices: int64")
  .Output("out_values: T")
  .Output("out_shape: int64")
  .Attr("strides: list(int)")
  .Attr("padding: string")
  .Attr("filter_dim: int = 3")
  .Attr("approx: bool = true");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T>
class SparseTensorSparseKernelDenseConvKD : public OpKernel {
 public:
  explicit SparseTensorSparseKernelDenseConvKD(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_dim", &filter_dim));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("approx", &approx));
  }

  void Compute(OpKernelContext* context) override {

    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
    OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
    OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
    auto in_ind = in_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();
    auto f_ind = filter_indices->matrix<int64>(); //filters, channels, kernel_depth, kernel_height, kernel_width TODO: other cases?
    auto f_vals = filter_values->flat<T>();
    auto f_sh = filter_shape->flat<int64>();

    std::vector<std::vector<int64> > output_keys; //stores the values for the output tensor
    std::vector<T> output_values;
    std::vector<int64> out_shape;

    sparseCuboidConvKD(in_ind, in_vals, in_sh, f_ind, f_vals, f_sh, stride_, filter_dim, output_keys, output_values, out_shape, padding, approx);

    // Create an output tensor
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
    TensorShape out_ind_shape = {(int64) output_keys.size(), (int64) in_ind.dimension(1)};
    TensorShape out_val_shape = {(int64) output_keys.size()};
    TensorShape out_sh_shape = {(int64) in_ind.dimension(1)};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &sparse_shape));

    auto out_ind = sparse_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto out_vals = sparse_values->flat<T>();
    auto out_sh = sparse_shape->flat<int64>();

    auto out_ind_ptr = &out_ind; auto out_vals_ptr = &out_vals; auto output_keys_ptr = &output_keys; auto output_values_ptr = &output_values;
#pragma omp parallel for firstprivate(output_keys_ptr, output_values_ptr, out_vals_ptr, out_ind_ptr)
    for(auto i = 0; i < (*output_keys_ptr).size(); ++i){
        for(int64 j = 0; j < (*output_keys_ptr)[i].size(); ++j){
          (*out_ind_ptr)(i,j) = (*output_keys_ptr)[i][j];
        }
        (*out_vals_ptr)(i) = (*output_values_ptr)[i];
    }
    for(int64 idx = 0; idx < in_ind.dimension(1); ++idx){
        out_sh(idx) = out_shape[idx];
    }
  }

 private:
  std::vector<int32> stride_;
  int32 filter_dim;
  std::string padding;
  bool approx;
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConvKD").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
                          SparseTensorSparseKernelDenseConvKD<CPUDevice, type>);

REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);
#undef REGISTER_CPU
