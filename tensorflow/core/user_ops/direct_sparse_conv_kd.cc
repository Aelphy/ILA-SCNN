#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#if GOOGLE_CUDA
#include "direct_sparse_conv_kd_gpu.h"
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


/*REGISTER_OP("DirectSparseConvKD")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values: T")
  .Input("in_shape: Tindices")
  .Input("filter_indices: Tindices")
  .Input("filter_values: T")
  .Input("filter_shape: Tindices")
  .Output("out_indices: Tindices")
  .Output("out_values: T")
  .Output("out_shape: Tindices")
  .Attr("strides: list(int)")
  .Attr("padding: string")
  .Attr("filter_dim: int = 3");
*/
REGISTER_OP("DirectSparseApproxConvKD")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values: T")
  .Input("in_shape: Tindices")
  .Input("filter_indices: Tindices")
  .Input("filter_values: T")
  .Input("filter_shape: Tindices")
  .Output("out_indices: Tindices")
  .Output("out_values: T")
  .Output("out_shape: Tindices")
  .Output("data_count: int32")
  .Attr("strides: list(int)")
  .Attr("padding: string")
  .Attr("filter_dim: int = 3")
  .Attr("max_density: float = 1");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename Tindices, typename FunctorT>
class DirectSparseConvKD : public OpKernel {
 public:
 explicit DirectSparseConvKD(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("filter_dim", &filter_dim));
    OP_REQUIRES(context, stride_.size() >= 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "at least specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("max_density", &max_density));
  }

  void Compute(OpKernelContext* context) override {
    //functor requires kernel context since output shape is not known befor computing results
    FunctorT()(context, stride_, padding, max_density);
  }

 private:
  std::vector<int32> stride_;
  Tindices filter_dim;
  std::string padding;
  float max_density;
};

#if GOOGLE_CUDA
#define REGISTER_GPU_TYPE(type, indice_type, dim)      \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseApproxConvKD").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConvKD<indice_type, functor::ApproxDirectSparseConvFunctor<GPUDevice, type, indice_type, dim> >);    //           \
//  REGISTER_KERNEL_BUILDER(Name("DirectSparseConvKD").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
//                          DirectSparseConvKD<indice_type, functor::DirectSparseConvFunctor<GPUDevice, type, indice_type> >);


#define REGISTER_GPU_TYPE_(type, indice_type) \
  REGISTER_GPU_TYPE(type, indice_type, 5);

#define REGISTER_GPU_ALL(type) \
  REGISTER_GPU_TYPE_(type, int64); \
  REGISTER_GPU_TYPE_(type, int32);


REGISTER_GPU_ALL(float);
//REGISTER_GPU_ALL(double);
//REGISTER_GPU_ALL(int32);
//REGISTER_GPU_ALL(complex64);
//REGISTER_GPU_ALL(complex128);
#undef REGISTER_GPU_TYPE
#undef REGISTER_GPU_TYPE_
#undef REGISTER_GPU_ALL
#endif //GOOGLE_CUDA

