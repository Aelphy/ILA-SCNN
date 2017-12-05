#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_cwise_biased_reg_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {

template <typename dtype, typename itype, int data_dimension, int power> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
biased_cwise_reg(CudaLaunchConfig config, const itype* __restrict__ f_id_ptr, const dtype* __restrict__ f_val_ptr, 
  const itype* __restrict__ f_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/, const int* __restrict__ f_mapping_ptr,
  const dtype* __restrict__ bias, const dtype* __restrict__ scale, dtype* loss)
{
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    auto in_id = f_id_ptr[x];
    index_KDto1D_<itype, data_dimension>(&id_kd[0], f_shape_ptr, &in_id);
    auto out_channel = id_kd[data_dimension - 1];
    auto out_v = scale[0] * pow(f_val_ptr[x] + bias[out_channel], power); //TODO: power as template parameter (L1, L2)
    atomicAdd(&(loss[out_channel]), out_v);
  }
}

namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension, int power>
void DirectSparseCwiseBiasedRegFunctor<DeviceT, T, IndiceT, data_dimension, power>::operator()(OpKernelContext* context) const {
  const Tensor *f_indices, *f_values, *f_shape, *f_block_ptr_t, *bias, *scale;
  OP_REQUIRES_OK(context, context->input("f_indices", &f_indices));
  OP_REQUIRES_OK(context, context->input("f_values", &f_values));
  OP_REQUIRES_OK(context, context->input("f_shape", &f_shape));
  OP_REQUIRES_OK(context, context->input("f_channel_mapping", &f_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("bias", &bias));
  OP_REQUIRES_OK(context, context->input("scale", &scale));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto f_sh = f_shape->flat<IndiceT>();
  auto f_ind = f_indices->flat<IndiceT>();
  auto f_val = f_values->flat<T>();
  auto f_mapping = f_block_ptr_t->flat<int>();
  auto b = bias->flat<T>();
  auto s = scale->flat<T>();
  int filter_weight_count;
  cudaMemcpy(&filter_weight_count, f_mapping.data() + f_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  IndiceT out_channel_count;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);

	Tensor *out_loss;
	TensorShape out_loss_shape = {(IndiceT) out_channel_count};
	OP_REQUIRES_OK(context, context->allocate_output("out_values", out_loss_shape, &out_loss));
  auto o_loss = out_loss->flat<T>();
  cudaMemset(o_loss.data(), 0, out_channel_count * sizeof(T));
  if(filter_weight_count > 0){
    CudaLaunchConfig config = GetCudaLaunchConfig(filter_weight_count, d);
    biased_cwise_reg<T, IndiceT, data_dimension, power><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, f_ind.data(), f_val.data(), f_sh.data(), f_mapping.data(), 
        b.data(), s.data(), o_loss.data());
  }
}


}  // end namespace functor

// Instantiate the GPU implementation for float.
//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type, dim, power) \
 template struct functor::DirectSparseCwiseBiasedRegFunctor<GPUDevice, type, indice_type, dim, power>;
#define INIT_GPU_ALL(type, dim)    \
  INIT_GPU_TYPE(type, int64, dim, 1); \
  INIT_GPU_TYPE(type, int32, dim, 1); \
  INIT_GPU_TYPE(type, int64, dim, 2); \
  INIT_GPU_TYPE(type, int32, dim, 2);
#define INIT_GPU_ALL_(type)    \
  INIT_GPU_ALL(type, 5);

INIT_GPU_ALL_(float);
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
#undef INIT_GPU_ALL_
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
