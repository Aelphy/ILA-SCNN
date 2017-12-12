#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#if GOOGLE_CUDA
#include "direct_sparse_cwise_biased_reg_gpu.h"
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


REGISTER_OP("DirectSparseChannelwiseBiasedL2Regularization")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("f_indices: Tindices")
  .Input("f_values: T")
  .Input("f_shape: Tindices")
  .Input("f_channel_mapping: int32")
  .Input("scale: T")
  .Input("bias: T")
  .Output("out_values: T")
  .Attr("dim: int = 5")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    int sparse_entry_count = 1; //TODO: not 1 but number out channels
    int dim;
    TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));
    std::vector<::tensorflow::shape_inference::DimensionHandle> sparse_output_shape_dims;
    sparse_output_shape_dims.push_back(c->MakeDim(1));
    c->set_output(0, c->MakeShape(sparse_output_shape_dims));
    return ::tensorflow::Status::OK();
  });

#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename T, typename Tindices, template<typename, typename, typename, int, int> class FunctorT>
class DirectSparseChannelwiseBiasedL2Regularization : public OpKernel {
 public:
 explicit DirectSparseChannelwiseBiasedL2Regularization(OpKernelConstruction* context) : OpKernel(context) 
 {
   OP_REQUIRES_OK(context, context->GetAttr("dim", &dim));  
 }

  void Compute(OpKernelContext* context) override {
    //TODO: more dimensions
    if(dim == 5){
      FunctorT<GPUDevice, T, Tindices, 5, 2>()(context);
    }
  }

 private:
   int dim;
};

#if GOOGLE_CUDA
#define REGISTER_GPU_TYPE(type, indice_type)      \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseChannelwiseBiasedL2Regularization").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseChannelwiseBiasedL2Regularization<type, indice_type, functor::DirectSparseCwiseBiasedRegFunctor>);

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
