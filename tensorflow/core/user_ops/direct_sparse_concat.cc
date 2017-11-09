#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#if GOOGLE_CUDA
#include "direct_sparse_concat_gpu.h"
#endif //GOOGLE_CUDA


/** DirectSparseConvKD
  * \ingroup CXX11_NeuralNetworks_Module
  * 
  * \brief Applies a 3D convolution over a multichannel input voxel block.
  * 
  * The input parameter is expected to be a tensor with a rank of 4 or more (channels, depth, height, width, and optionally others).
  * The kernel parameter is expected to be a 5D tensor (filters, channels, kernel_depth, kernel_height, kernel_width).
  * 
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be filters, depth, height, width (and others if applicable).
  */

//both inputs have same in_block_channel_mapping!
REGISTER_OP("DirectSparseConcat")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values1: T")
  .Input("in_values2: T")
  .Input("in_shape: Tindices")
  .Input("in_block_channel_mapping: int32")
  .Output("out_indices: Tindices")
  .Output("out_values: T")
  .Output("out_shape: Tindices")
  .Output("out_block_channel_mapping: int32")
  .Attr("dim: int = 5");

//both inputs have same in_block_channel_mapping!
REGISTER_OP("DirectSparseConcatBackprop")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values1: T")
  .Input("in_values2: T")
  .Input("in_shape: Tindices")
  .Input("in_block_channel_mapping: int32")
  .Input("out_indices: Tindices")
  .Input("out_values: T")
  .Input("out_shape: Tindices")
  .Input("out_block_channel_mapping: int32")
  .Input("grads: T")
  .Output("input_grads1: T")
  .Output("input_grads2: T")
  .Attr("dim: int = 5");

#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename T, typename Tindices, template<typename, typename, typename, int> class FunctorT>
class DirectSparseConcat : public OpKernel {
 public:
 explicit DirectSparseConcat(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dim", &dim));
  }

  void Compute(OpKernelContext* context) override {
    //functor requires kernel context since output shape is not known befor computing results
    if(dim == 5){
      FunctorT<GPUDevice, T, Tindices, 5>()(context);
    } //TODO: add more dimensions
  }

 private:
  int dim;
};

#if GOOGLE_CUDA
#define REGISTER_GPU_TYPE(type, indice_type)      \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseConcat").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConcat<type, indice_type, functor::DirectSparseConcatFunctor>); \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseConcatBackprop").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConcat<type, indice_type, functor::DirectSparseConcatBackPropFunctor>);

#define REGISTER_GPU_ALL(type) \
  REGISTER_GPU_TYPE(type, int64); \
  REGISTER_GPU_TYPE(type, int32);


REGISTER_GPU_ALL(float);
//REGISTER_GPU_ALL(double);
//REGISTER_GPU_ALL(int32);
//REGISTER_GPU_ALL(complex64);
//REGISTER_GPU_ALL(complex128);
#undef REGISTER_GPU_TYPE
#undef REGISTER_GPU_ALL
#endif //GOOGLE_CUDA

