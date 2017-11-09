#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_concat_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {

template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_values1(CudaLaunchConfig config, const itype* __restrict__ in_id_ptr, const dtype* __restrict__ in_val_ptr, 
  const itype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/, itype* out_ind_ptr, dtype* out_val_ptr)
{
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    out_val_ptr[x] = in_val_ptr[x];
    out_ind_ptr[x] = in_id_ptr[x];
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_values2(CudaLaunchConfig config, const itype* __restrict__ in_id_ptr, const dtype* __restrict__ in_val_ptr, 
  const itype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/, const itype* __restrict__ out_shape_ptr,
  itype* out_ind_ptr, dtype* out_val_ptr, int offset)
{
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype in_index_1d = in_id_ptr[x];
    dtype in_val = in_val_ptr[x];
    index_1DtoKD<itype, data_dimension>(0, in_index_1d, in_shape_ptr, &id_kd[0]);
    id_kd[data_dimension - 1] = id_kd[data_dimension - 1] + in_shape_ptr[data_dimension - 1];
    itype out_ind = -1;
    index_KDto1D_<itype, data_dimension>(&id_kd[0], out_shape_ptr, &out_ind);
    out_val_ptr[x + offset] = in_val_ptr[x];
    out_ind_ptr[x + offset] = out_ind;
  }
}


template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
vector_add(CudaLaunchConfig config, dtype*__restrict__ values, dtype add_value){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    values[x] += add_value; 
  }
}

namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConcatFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  const Tensor *in_indices, *in_values1, *in_values2, *in_shape, *in_block_ptr_t;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values1", &in_values1));
  OP_REQUIRES_OK(context, context->input("in_values2", &in_values2));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_ptr_t));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val1 = in_values1->flat<T>();
  auto i_val2 = in_values2->flat<T>();
  auto i_mapping = in_block_ptr_t->flat<int>();
  int data_entry_count;
  cudaMemcpy(&data_entry_count, i_mapping.data() + i_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);

  if(i_val1.dimension(0) != i_val2.dimension(0)){
    LOG(ERROR) << "dimensions of value tensors do not match; sizes are " << i_val1.dimension(0) << " " << i_val2.dimension(0) << std::endl;
  }
	Tensor *out_shape, *out_block_channel_mapping;
	Tensor *out_indices, *out_values;
	TensorShape out_sh_shape = {(IndiceT) data_dimension};
	TensorShape out_bcm_shape = {(IndiceT) 2 * i_mapping.dimension(0)}; 
	TensorShape out_val_shape = {(IndiceT) 2 * i_val1.dimension(0)}; 
	TensorShape out_ind_shape = {(IndiceT) 2 * i_ind.dimension(0)}; 
	OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
	OP_REQUIRES_OK(context, context->allocate_output("out_block_channel_mapping", out_bcm_shape, &out_block_channel_mapping));
	OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
	OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_mapping = out_block_channel_mapping->flat<int>();
  cudaMemset(o_mapping.data(), 0, o_mapping.dimension(0) * sizeof(int));
  if(data_entry_count > 0){
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    //fill first half
    concat_values1<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), i_val1.data(), i_sh.data(), o_ind.data(), o_val.data());
    //output shape
    std::vector<IndiceT> cpu_shape(data_dimension);
    cudaMemcpy(&cpu_shape[0], i_sh.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    cpu_shape[data_dimension - 1] =  cpu_shape[data_dimension - 1] * 2;
    cudaMemcpy(o_sh.data(), &cpu_shape[0], (data_dimension) * sizeof(IndiceT), cudaMemcpyHostToDevice);

    concat_values2<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), i_val2.data(), i_sh.data(), o_sh.data(), o_ind.data(), o_val.data(), data_entry_count);
    
    cudaMemcpy(o_mapping.data(), i_mapping.data(), i_mapping.dimension(0) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(o_mapping.data() + data_entry_count, i_mapping.data(), i_mapping.dimension(0) * sizeof(int), cudaMemcpyDeviceToDevice);
    
    vector_add<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, o_mapping.data() + data_entry_count, data_entry_count);
  }
}


template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConcatBackPropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  const Tensor  *in_indices, *in_values1, *in_values2, *in_shape, *in_block_ptr_t, 
                *out_indices, *out_values, *out_shape, *out_block_ptr_t, *gradients;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values1", &in_values1));
  OP_REQUIRES_OK(context, context->input("in_values2", &in_values2));
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
  auto i_val1 = in_values1->flat<T>();
  auto i_val2 = in_values2->flat<T>();
  auto i_mapping = in_block_ptr_t->flat<int>();
  const int* input_block_mapping = i_mapping.data();
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_mapping = out_block_ptr_t->flat<int>();
  const int* output_block_mapping = o_mapping.data();
  auto grads = gradients->flat<T>(); 
  int data_entry_count;
  cudaMemcpy(&data_entry_count, i_mapping.data() + i_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
}

}  // end namespace functor

// Instantiate the GPU implementation for float.
//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseConcatFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseConcatBackPropFunctor<GPUDevice, type, indice_type, dim>;
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
