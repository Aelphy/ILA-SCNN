#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status.h"
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


REGISTER_OP("DirectSparseConvKD")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values: T")
  .Input("in_shape: Tindices")
  .Input("in_block_channel_mapping: int32")
  .Input("filter_indices: Tindices")
  .Input("filter_values: T")
  .Input("filter_shape: Tindices")
  .Input("filter_channel_mapping: int32")
  .Input("bias: T")
  .Output("out_indices: Tindices")
  .Output("out_values: T")
  .Output("out_shape: Tindices")
  .Output("out_block_channel_mapping: int32")
  .Output("out_channel_densities: float32")
  .Attr("strides: list(int)")
  .Attr("padding: string")
  .Attr("dense_entry_count: int")
  .Attr("dim: int = 5")
  .Attr("max_density: float = 1")
  .Attr("filter_type: string = 'K-ABS'")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    float max_density;
    TF_RETURN_IF_ERROR(c->GetAttr("max_density", &max_density));

    int dense_entry_count;
    TF_RETURN_IF_ERROR(c->GetAttr("dense_entry_count", &dense_entry_count));

    int dim;
    TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));

    std::vector<::tensorflow::shape_inference::DimensionHandle> sparse_output_shape_dims;
    sparse_output_shape_dims.push_back(c->MakeDim(ceil(dense_entry_count * max_density)));

    std::vector<::tensorflow::shape_inference::DimensionHandle> output_shape_dims;
    output_shape_dims.push_back(c->MakeDim(dim));

    c->set_output(0, c->MakeShape(sparse_output_shape_dims));
    c->set_output(1, c->MakeShape(sparse_output_shape_dims));
    c->set_output(2, c->MakeShape(output_shape_dims));
    c->set_output(3, c->UnknownShape());
    c->set_output(4, c->UnknownShape());

    return ::tensorflow::Status::OK();
  });

REGISTER_OP("DirectSparseConvKDBackprop")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values: T")
  .Input("in_shape: Tindices")
  .Input("in_block_channel_mapping: int32")
  .Input("filter_indices: Tindices")
  .Input("filter_values: T")
  .Input("filter_shape: Tindices")
  .Input("filter_channel_mapping: int32")
  .Input("out_indices: Tindices")
  .Input("out_values: T")
  .Input("out_shape: Tindices")
  .Input("out_block_channel_mapping: int32")
  .Input("grads: T")
  .Output("input_grads: T")
  .Output("filter_grads: T")
  .Output("bias_grads: T")
  .Attr("strides: list(int)")
  .Attr("padding: string")
  .Attr("dense_entry_count: int = 1")
  .Attr("dim: int = 5")
  .Attr("max_density: float = 1")
  .Attr("filter_type: string = 'K-ABS'");

#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename T, typename Tindices, template<typename, typename, typename, int> class FunctorT>
class DirectSparseConvKD : public OpKernel {
 public:
 explicit DirectSparseConvKD(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES_OK(context, context->GetAttr("dim", &dim));
    OP_REQUIRES(context, stride_.size() >= 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "at least specify 4 dimensions"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("max_density", &max_density));
    OP_REQUIRES_OK(context, context->GetAttr("filter_type", &filter_type));
    OP_REQUIRES_OK(context, context->GetAttr("dense_entry_count", &dense_entry_count));
  }

  void Compute(OpKernelContext* context) override {
    //functor requires kernel context since output shape is not known befor computing results
    if(dim == 5){
      FunctorT<GPUDevice, T, Tindices, 5>()(context, stride_, padding, max_density, filter_type);
    } //TODO: add more dimensions
  }

 private:
  std::vector<int32> stride_;
  int dim;
  std::string padding;
  std::string filter_type;
  float max_density;
  int dense_entry_count;
};

#if GOOGLE_CUDA
#define REGISTER_GPU_TYPE(type, indice_type)      \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseConvKD").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConvKD<type, indice_type, functor::DirectSparseConvFunctor>); \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseConvKDBackprop").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConvKD<type, indice_type, functor::DirectSparseConvBackPropFunctor>);

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
