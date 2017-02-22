#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "sparse_tensor_sparse_kernel_dense_conv_3d.h"


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



//TODO: How do I use REGISTER_OP with parameter T?
//  .Attr("T: {float, double, int32, complex64, complex128}")
REGISTER_OP("SparseTensorSparseKernelDenseConv3D")
  .Attr("T: {float}")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Output("sparse_indices: int64")
  .Output("sparse_values: T")
  .Output("sparse_shape: int64")
  .Attr("strides: list(int) >= 5");
//  .Output("debug_output: string")





#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Device, typename T>
class SparseTensorSparseKernelDenseConv3D : public OpKernel {
 public:
  explicit SparseTensorSparseKernelDenseConv3D(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
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

    std::map<std::vector<int64>, T> output_map; //stores the values for the output tensor
    std::vector<int64> out_shape;

    sparseCuboidConv3D(in_ind, in_vals, in_sh, f_ind, f_vals, f_sh, stride_, output_map, out_shape);

    // Create an output tensor
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
    TensorShape out_ind_shape = {(int64) output_map.size(), (int64) in_ind.dimension(1)};
    TensorShape out_val_shape = {(int64) output_map.size()};
    TensorShape out_sh_shape = {(int64) in_ind.dimension(1)};
    OP_REQUIRES_OK(context, context->allocate_output("sparse_indices", out_ind_shape, &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("sparse_values", out_val_shape, &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("sparse_shape", out_sh_shape, &sparse_shape));

    auto out_ind = sparse_indices->matrix<int64>(); //channels, depth, height, width, optionally others TODO: other cases?
    auto out_vals = sparse_values->flat<T>();
    auto out_sh = sparse_shape->flat<int64>();

    int64 idx = 0;
    for(auto it = output_map.begin(); it != output_map.end(); ++it, idx++){
        const std::vector<int64> &indice = it->first;
        for(int64 j = 0; j < indice.size(); ++j){
          out_ind(idx,j) = indice[j];
        }
        out_vals(idx) = it->second;
    }
    for(int64 idx = 0; idx < in_ind.dimension(1); ++idx){
        out_sh(idx) = out_shape[idx];
    }


    //debug output
/*    Tensor* debug_ = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &debug_));
    auto output_ = debug_->scalar<string>();
    //std::stringstream debug; debug << "DEBUG OUTPUT:" << std::endl; 
    //std::string deb_string = debug.str();
    output_() = "test";
*/  
  }

 private:
  std::vector<int32> stride_;
};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConv3D").Device(DEVICE_CPU), SparseTensorSparseKernelDenseConv3D<CPUDevice, type>);

REGISTER_CPU(float);
//REGISTER_CPU(double);
//REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);

