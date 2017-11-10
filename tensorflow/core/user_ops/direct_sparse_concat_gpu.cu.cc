#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_concat_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {

template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_values(CudaLaunchConfig config, const itype* __restrict__ in_id_ptr, const dtype* __restrict__ in_val1_ptr, 
  const dtype* __restrict__ in_val2_ptr, const itype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
  const itype* __restrict__ out_shape_ptr,  const int* __restrict__ in_mapping_ptr, const int* __restrict__ out_mapping_ptr,
  itype* out_ind_ptr, dtype* out_val_ptr)
{
  int in_channel_count = in_shape_ptr[data_dimension - 1];
  int out_channel_count = 2 * in_channel_count;
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype in_index_1d = in_id_ptr[x];
    index_1DtoKD<itype, data_dimension>(0, in_index_1d, in_shape_ptr, &id_kd[0]);
    itype channel1 = id_kd[data_dimension - 1];
    itype channel2 = in_channel_count + channel1;
    itype batch = id_kd[data_dimension - 1];
    int in_channel_start_ptr = in_mapping_ptr[batch * in_channel_count + channel1];
    int offset = x - in_channel_start_ptr;
    int out_channel_start1_ptr = out_mapping_ptr[batch * out_channel_count + channel1];
    int out_channel_start2_ptr = out_mapping_ptr[batch * out_channel_count + channel2];
    out_val_ptr[out_channel_start1_ptr + offset] = in_val1_ptr[x];
    out_ind_ptr[out_channel_start1_ptr + offset] = in_id_ptr[x];
    itype in_id2;
    id_kd[data_dimension - 1] = id_kd[data_dimension - 1] + in_channel_count;
    index_KDto1D_<itype, data_dimension>(&id_kd[0], out_shape_ptr, &in_id2);
    out_val_ptr[out_channel_start2_ptr + offset] = in_val2_ptr[x];
    out_ind_ptr[out_channel_start2_ptr + offset] = in_id2;
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_backprop_values(CudaLaunchConfig config, const itype* __restrict__ out_id_ptr, const itype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/, const itype* __restrict__ out_shape_ptr, const int* __restrict__ in_mapping_ptr, const int* __restrict__ out_mapping_ptr,
  const dtype* __restrict__ in_grads, dtype* out_grads1, dtype* out_grads2)
{
  int in_channel_count = in_shape_ptr[data_dimension - 1];
  int out_channel_count = 2 * in_channel_count;
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype out_index_1d = out_id_ptr[x];
    index_1DtoKD<itype, data_dimension>(0, out_index_1d, out_shape_ptr, &id_kd[0]);
    itype channel = id_kd[data_dimension - 1];
    itype bp_channel = channel;
    bool bp2 = false;
    if(channel >= in_channel_count){
      bp_channel -= in_channel_count;
      bp2 = true;
    }
    itype batch = id_kd[data_dimension - 1];
    int in_channel_start_ptr = in_mapping_ptr[batch * in_channel_count + bp_channel];
    int out_channel_start_ptr = out_mapping_ptr[batch * out_channel_count + channel];
    int offset = x - out_channel_start_ptr;
    if(bp2 != true){
      out_grads1[in_channel_start_ptr + offset] = in_grads[x];
    } else {
      out_grads2[in_channel_start_ptr + offset] = in_grads[x];
    }
  }
}

template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_out_mapping(CudaLaunchConfig config, const dtype*__restrict__ in_mapping, dtype in_channel_count, dtype in_batch_count, dtype* out_channel_mapping){
  int out_channel_count = 2 * in_channel_count;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    int channel =  x % in_channel_count;
    int channel2 = out_channel_count + channel; //concatinated channels
    int batch = (x - channel) / in_channel_count;
    int in_chanel_ptr = in_mapping[batch * in_channel_count + channel];
    int in_batch_start_ptr = in_mapping[batch * in_channel_count];
    int in_batch_end_ptr = in_mapping[(batch + 1) * in_channel_count];
    int batch_size = in_batch_end_ptr - in_batch_start_ptr;
    int channel_offset = in_chanel_ptr - in_batch_start_ptr;
    out_channel_mapping[batch * out_channel_count + channel] = 2 * in_batch_start_ptr + channel_offset;
    out_channel_mapping[batch * out_channel_count + channel2] = 2 * in_batch_start_ptr + batch_size + channel_offset;
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
    std::vector<IndiceT> cpu_shape(data_dimension);
    cudaMemcpy(&cpu_shape[0], i_sh.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    int in_channel_count = cpu_shape[data_dimension - 1];
    int in_batch_count = cpu_shape[0];
    cpu_shape[data_dimension - 1] = cpu_shape[data_dimension - 1] * 2;
    cudaMemcpy(o_sh.data(), &cpu_shape[0], (data_dimension) * sizeof(IndiceT), cudaMemcpyHostToDevice);
    CudaLaunchConfig configm = GetCudaLaunchConfig(i_mapping.dimension(0), d);
    compute_out_mapping<<<configm.block_count, configm.thread_per_block, 0, d.stream()>>>(configm, i_mapping.data(), in_channel_count, in_batch_count, o_mapping.data());
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    concat_values<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), i_val1.data(), i_val2.data(), i_sh.data(), o_sh.data(), i_mapping.data(), o_mapping.data(), o_ind.data(), o_val.data());
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
  int out_entry_count;
  cudaMemcpy(&out_entry_count, o_mapping.data() + o_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
	Tensor *backprop1, *backprop2;
	TensorShape out_bp1_shape = {(IndiceT) i_val1.dimension(0)}; 
	TensorShape out_bp2_shape = {(IndiceT) i_val2.dimension(0)}; 
	OP_REQUIRES_OK(context, context->allocate_output("backprop1", out_bp1_shape, &backprop1));
	OP_REQUIRES_OK(context, context->allocate_output("backprop2", out_bp2_shape, &backprop2));
  auto bp1 = backprop1->flat<T>();
  auto bp2 = backprop2->flat<T>();
  cudaMemset(bp1.data(), 0, bp1.dimension(0) * sizeof(T));
  cudaMemset(bp2.data(), 0, bp2.dimension(0) * sizeof(T));
  if(out_entry_count > 0){
    CudaLaunchConfig config = GetCudaLaunchConfig(out_entry_count, d);
    concat_backprop_values<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, o_ind.data(), i_sh.data(), o_sh.data(), i_mapping.data(), o_mapping.data(), grads.data(), bp1.data(), bp2.data());
  }
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
