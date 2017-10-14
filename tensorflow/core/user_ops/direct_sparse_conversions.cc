#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#if GOOGLE_CUDA
#include "direct_sparse_conversions_gpu.h"
#endif //GOOGLE_CUDA


REGISTER_OP("DirectSparseDataConversion")
  .Attr("T: realnumbertype")
  .Attr("Tindices: {int32, int64}")
  .Input("in_indices: Tindices")
  .Input("in_values: T")
  .Input("in_shape: Tindices")
  .Output("out_indices: Tindices")
  .Output("out_block_ptr: Tindices")
  .Output("out_values: T")
  .Output("out_shape: Tindices")
  .Attr("dim: int=5");


#include "tensorflow/core/framework/op_kernel.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

template <typename T, typename Tindices, template<typename, typename, typename, int> class FunctorT>
class DirectSparseConversion : public OpKernel {
 public:
 explicit DirectSparseConversion(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dim", &dim));
  }

  void Compute(OpKernelContext* context) override {
    //functor requires kernel context since output shape is not known befor computing results
    if(dim == 5){
      FunctorT<GPUDevice, T, Tindices, 5>()(context);
    } //TODO: add more dimensions
  }

 private:
  Tindices dim;
};

#if GOOGLE_CUDA
#define REGISTER_GPU_TYPE(type, indice_type, dim)      \
  REGISTER_KERNEL_BUILDER(Name("DirectSparseDataConversion").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConversion<type, indice_type, functor::DirectSparseDataConversionFunctor>); //\
  REGISTER_KERNEL_BUILDER(Name("DirectSparseDataConversion").Device(DEVICE_GPU).TypeConstraint<type>("T").TypeConstraint<indice_type>("Tindices"), \
                          DirectSparseConversion<type, indice_type, functor::DirectSparseFilterConversionFunctor>);


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

