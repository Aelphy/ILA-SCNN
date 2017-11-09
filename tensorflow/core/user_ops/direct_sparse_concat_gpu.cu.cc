#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_conv_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {


namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConcatFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  clock_t t;
  const Tensor *in_indices, *in_values, *in_shape, *in_block_ptr_t, *data_count_t, *filter_indices, *filter_values, *filter_shape, *filter_channel_mapping_t;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_ptr_t));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto i_mapping = in_block_ptr_t->flat<int>();
  int data_entry_count;
  cudaMemcpy(&data_entry_count, i_mapping.data() + i_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
}


template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConcatBackPropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  const Tensor  *in_indices, *in_values, *in_shape, *in_block_ptr_t, *out_indices, *out_values, *out_shape, *out_block_ptr_t, 
                *data_count_t, *filter_indices, *filter_values, *filter_shape, *filter_channel_mapping_t, *gradients;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
  OP_REQUIRES_OK(context, context->input("out_values", &out_values));
  OP_REQUIRES_OK(context, context->input("out_shape", &out_shape));
  OP_REQUIRES_OK(context, context->input("out_block_channel_mapping", &out_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("grads", &gradients));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto i_mapping = in_block_ptr_t->flat<int>();
  const int* input_block_mapping = i_mapping.data();
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_mapping = out_block_ptr_t->flat<int>();
  const int* output_block_mapping = o_mapping.data();
  auto grads = gradients->flat<T>(); 
  int data_entry_count, filter_weight_count, output_entry_count;
  cudaMemcpy(&data_entry_count, i_mapping.data() + i_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
}

}  // end namespace functor

// Instantiate the GPU implementation for float.
//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseConvFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseConvBackPropFunctor<GPUDevice, type, indice_type, dim>;
#define INIT_GPU_ALL(type, dim)    \
  INIT_GPU_TYPE(type, int64, dim); \
  INIT_GPU_TYPE(type, int32, dim);
#define INIT_GPU_ALL_(type)    \
  INIT_GPU_ALL(type, 5);

INIT_GPU_ALL_(float);
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
#undef INIT_GPU_ALL_
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
