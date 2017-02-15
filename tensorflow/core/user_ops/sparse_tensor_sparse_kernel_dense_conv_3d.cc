#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"


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
  .Attr("T: {int32}")
  .Input("in_indices: int64")
  .Input("in_values: T")
  .Input("in_shape: int64")
  .Input("filter_indices: int64")
  .Input("filter_values: T")
  .Input("filter_shape: int64")
  .Output("debug: string")
  .Attr("strides: list(int) >= 5");
//  .Output("sparse_indices: int64")
//  .Output("sparse_values: T")
//  .Output("sparse_shape: int64")



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
    std::stringstream debug; 
    //get input data
    const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
    OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
    OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
    auto in_ind = in_indices->matrix<int64>();
    auto in_vals = in_values->flat<T>();
    auto in_sh = in_shape->flat<int64>();

    //generate map with valid indices (those indices of input, which are not affected by stride):
    std::map<int64, std::pair<int64, int64> > valid_indices;
    for(int64 i = 0; i < in_ind.dimension(1); ++i){

    }










    //auto output = sparse_values->flat<int32>();


    /*OP_REQUIRES_OK(context, context->GetAttr("strides", &stride));

    for(size_t i = 0; i < input_indices.size(); ++i){
      if(index_is_unaffected_by_stride(input_indices[i], stride)){
          update_indices, update_values = dense_filter_update_output(input_indices[i], input_values[i], filter);
          SparseAddOp(update_indices, update_values)
      }
    }*/


/*
    // Create an output tensor
    Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
    OP_REQUIRES_OK(context, context->allocate_output("sparse_indices", in_values->shape(), &sparse_indices));
    OP_REQUIRES_OK(context, context->allocate_output("sparse_values", in_values->shape(), &sparse_values));
    OP_REQUIRES_OK(context, context->allocate_output("sparse_shape", in_values->shape(), &sparse_shape));
*/


    debug << "in ind: " << in_ind << std::endl << "in vals" << in_vals << std::endl << "in shape: " << in_sh << std::endl;
    //debug output
    Tensor* debug_ = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &debug_));
    auto output = debug_->scalar<string>();
    output() = debug.str().c_str();
  }

 private:
  template<typename Matrix2DT>  
  inline bool index_is_unaffected_by_stride(const Matrix2DT& ids, const std::vector<int32>& stride) const {
    //assert(id.size() == stride.size());
    for(int32 i = 0; i < stride.size(); ++i){
      if(stride[i] > 0){
        if(((ids[i] + 1) % stride[i]) == 0){
          return false;
        }
      }
    }
    return true;
  }

  /*
  inline void TTypes<Index>::Matrix indices dense_filter_update_output(TTypes<Index>::Matrix& a_conv_filter, TTypes<Index>::Matrix& a_input_indices, ){

  }
  */
  std::vector<int32> stride_;

};

#define REGISTER_CPU(type)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorSparseKernelDenseConv3D").Device(DEVICE_CPU), SparseTensorSparseKernelDenseConv3D<CPUDevice, type>);

//REGISTER_CPU(float);
//REGISTER_CPU(double);
REGISTER_CPU(int32);
//REGISTER_CPU(complex64);
//REGISTER_CPU(complex128);

