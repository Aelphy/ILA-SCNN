#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_concat_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"


namespace tensorflow {

template <typename itype> __device__ __forceinline__ void  
index_mapping(
  const int in_id,
  const itype in_channel,
  const itype out_channel,
  const itype batch,
  const itype in_channel_count,
  const itype out_channel_count,
  const int* __restrict__ i_mapping_ptr,
  const int* __restrict__ o_mapping_ptr,
  int* out_id)
{
  int block_start_id = i_mapping_ptr[batch * in_channel_count + in_channel];
  int offset = in_id - block_start_id;
  *out_id = o_mapping_ptr[batch * out_channel_count + out_channel] + offset;
}

//perform first part of concationation
template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_values1(CudaLaunchConfig config, 
  const itype* __restrict__ in_id_ptr, 
  const dtype* __restrict__ in_val_ptr, 
  const itype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
  const itype* __restrict__ out_shape_ptr,
  const int* __restrict__ i_mapping_ptr,
  const int* __restrict__ o_mapping_ptr,
  itype* out_ind_ptr, 
  dtype* out_val_ptr)
{
  itype id_kd[data_dimension];
  int in_channel_count = in_shape_ptr[data_dimension - 1];
  int out_channel_count = out_shape_ptr[data_dimension - 1];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype in_index_1d = in_id_ptr[x];
    index_1DtoKD<itype, data_dimension>(0, in_index_1d, in_shape_ptr, &id_kd[0]);
    itype in_id;
    index_KDto1D_<itype, data_dimension>(&id_kd[0], out_shape_ptr, &in_id);
    int out_id;
    itype in_channel = id_kd[data_dimension - 1];
    itype out_channel = id_kd[data_dimension - 1];
    itype batch = id_kd[0];
    index_mapping<itype>(x, 
      in_channel, 
      out_channel, 
      batch, 
      in_channel_count,
      out_channel_count,
      i_mapping_ptr,
      o_mapping_ptr,
      &out_id);
    out_val_ptr[out_id] = in_val_ptr[x];
    out_ind_ptr[out_id] = in_id;
  }
}

//perform second part of concationation
template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_values2(CudaLaunchConfig config, 
  const itype* __restrict__ in_id_ptr, 
  const dtype* __restrict__ in_val_ptr, 
  const itype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
  const itype* __restrict__ out_shape_ptr,
  int channel_offset, 
  const int* __restrict__ i_mapping_ptr,
  const int* __restrict__ o_mapping_ptr,
  itype* out_ind_ptr, 
  dtype* out_val_ptr)
{
  itype id_kd[data_dimension];
  int in_channel_count = in_shape_ptr[data_dimension - 1];
  int out_channel_count = out_shape_ptr[data_dimension - 1];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype in_index_1d = in_id_ptr[x];
    index_1DtoKD<itype, data_dimension>(0, in_index_1d, in_shape_ptr, &id_kd[0]);
    itype in_channel = id_kd[data_dimension - 1];
    id_kd[data_dimension - 1 ] = id_kd[data_dimension - 1 ] + channel_offset;
    itype out_channel = id_kd[data_dimension - 1];
    itype in_id;
    index_KDto1D_<itype, data_dimension>(&id_kd[0], out_shape_ptr, &in_id);
    int out_id;
    itype batch = id_kd[0];
    index_mapping<itype>(x, 
      in_channel, 
      out_channel, 
      batch, 
      in_channel_count,
      out_channel_count,
      i_mapping_ptr,
      o_mapping_ptr,
      &out_id);
    out_val_ptr[out_id] = in_val_ptr[x];
    out_ind_ptr[out_id] = in_id;
  }
}


template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
concat_backprop_values(CudaLaunchConfig config, 
  const itype* __restrict__ out_id_ptr, 
  const itype* __restrict__ in_shape1_ptr /*[batch, dim1, ..., dimx, channel_nr]*/, 
  const itype* __restrict__ in_shape2_ptr /*[batch, dim1, ..., dimx, channel_nr]*/, 
  const itype* __restrict__ out_shape_ptr, 
  const int* __restrict__ i_mapping1_ptr, 
  const int* __restrict__ i_mapping2_ptr, 
  const int* __restrict__ o_mapping_ptr,
  const dtype* __restrict__ out_grads, 
  dtype* in_grads1, 
  dtype* in_grads2)
{
  int in_channel_count1 = in_shape1_ptr[data_dimension - 1];
  int in_channel_count2 = in_shape2_ptr[data_dimension - 1];
  int out_channel_count = in_channel_count1 + in_channel_count2;
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype out_index_1d = out_id_ptr[x];
    index_1DtoKD<itype, data_dimension>(0, out_index_1d, out_shape_ptr, &id_kd[0]);
    itype out_channel = id_kd[data_dimension - 1];
    itype batch = id_kd[data_dimension - 1];
    int in_id;
    if(out_channel < in_channel_count1){
      itype in_channel = out_channel;
      index_mapping<itype>(x, 
        out_channel, 
        in_channel, 
        batch, 
        out_channel_count,
        in_channel_count1,
        o_mapping_ptr,
        i_mapping1_ptr,
        &in_id);
      in_grads1[in_id] = out_grads[x];
    } else {
      itype in_channel = out_channel - in_channel_count1;
      index_mapping<itype>(x, 
        out_channel, 
        in_channel, 
        batch, 
        out_channel_count,
        in_channel_count2,
        o_mapping_ptr,
        i_mapping2_ptr,
        &in_id);
      in_grads2[in_id] = out_grads[x];
    }
  }
}

namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConcatFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  const Tensor *in_indices1, *in_values1, *in_shape1, *in_block_ptr_t1, *in_indices2, *in_values2, *in_shape2, *in_block_ptr_t2;
  OP_REQUIRES_OK(context, context->input("in_indices1", &in_indices1));
  OP_REQUIRES_OK(context, context->input("in_values1", &in_values1));
  OP_REQUIRES_OK(context, context->input("in_shape1", &in_shape1));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping1", &in_block_ptr_t1));
  OP_REQUIRES_OK(context, context->input("in_indices2", &in_indices2));
  OP_REQUIRES_OK(context, context->input("in_values2", &in_values2));
  OP_REQUIRES_OK(context, context->input("in_shape2", &in_shape2));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping2", &in_block_ptr_t2));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh1 = in_shape1->flat<IndiceT>();
  auto i_ind1 = in_indices1->flat<IndiceT>();
  auto i_val1 = in_values1->flat<T>();
  auto i_mapping1 = in_block_ptr_t1->flat<int>();
  auto i_sh2 = in_shape2->flat<IndiceT>();
  auto i_ind2 = in_indices2->flat<IndiceT>();
  auto i_val2 = in_values2->flat<T>();
  auto i_mapping2 = in_block_ptr_t2->flat<int>();
  int data_entry_count1;
  cudaMemcpy(&data_entry_count1, i_mapping1.data() + i_mapping1.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  int data_entry_count2;
  cudaMemcpy(&data_entry_count2, i_mapping2.data() + i_mapping2.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);

	Tensor *out_shape, *out_block_channel_mapping;
	Tensor *out_indices, *out_values;
	TensorShape out_sh_shape = {(IndiceT) data_dimension};
	TensorShape out_bcm_shape = {(IndiceT) i_mapping1.dimension(0) + i_mapping2.dimension(0) - 1}; 
	TensorShape out_val_shape = {(IndiceT) i_val1.dimension(0) + i_val2.dimension(0)}; 
	TensorShape out_ind_shape = {(IndiceT) i_ind1.dimension(0) + i_ind2.dimension(0)}; 
	OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
	OP_REQUIRES_OK(context, context->allocate_output("out_block_channel_mapping", out_bcm_shape, &out_block_channel_mapping));
	OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
	OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_mapping = out_block_channel_mapping->flat<int>();
  cudaMemset(o_mapping.data(), 0, o_mapping.dimension(0) * sizeof(int));
  cudaMemset(o_ind.data(), 0, o_ind.dimension(0) * sizeof(IndiceT));
  int in_channel_count1;
  if(data_entry_count1 > 0 || data_entry_count2 > 0){
    std::vector<IndiceT> cpu_shape1(data_dimension);
    cudaMemcpy(&cpu_shape1[0], i_sh1.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    std::vector<IndiceT> cpu_shape2(data_dimension);
    cudaMemcpy(&cpu_shape2[0], i_sh2.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    in_channel_count1 = cpu_shape1[data_dimension - 1];
    int in_channel_count2 = cpu_shape2[data_dimension - 1];
    int in_batch_count1 = cpu_shape1[0];
    int concat_channel_count = in_channel_count1 + in_channel_count2;
    for(size_t i = 0; i < data_dimension - 1; ++i){
      if(cpu_shape1[i] != cpu_shape2[i]){
        LOG(ERROR) << "Concatination: shapes of the tensors do not match" << std::endl;
        return;
      }
    }
    cpu_shape1[data_dimension - 1] = concat_channel_count;
    //TODO: compute on device
    cudaMemcpy(o_sh.data(), &cpu_shape1[0], (data_dimension) * sizeof(IndiceT), cudaMemcpyHostToDevice);
    std::vector<int> cpu_mapping1(i_mapping1.dimension(0));
    cudaMemcpy(&cpu_mapping1[0], i_mapping1.data(), i_mapping1.dimension(0) * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> cpu_mapping2(i_mapping2.dimension(0));
    cudaMemcpy(&cpu_mapping2[0], i_mapping2.data(), i_mapping2.dimension(0) * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> cpu_mapping_out(in_batch_count1 * concat_channel_count, 0);
    IndiceT index = 0;
    for(int i = 0; i < cpu_mapping1.size(); ++i){
      LOG(INFO) << cpu_mapping1[i] << " " << cpu_mapping2[i] << std::endl;
    }
    for(size_t i = 0; i < in_batch_count1; ++i){
      for(size_t j = 0; j < concat_channel_count; ++j){
        if(j < in_channel_count1){
          index += cpu_mapping1[i * in_channel_count1 + j + 1] - cpu_mapping1[i * in_channel_count1 + j];
        } else {
          size_t j2 = j - in_channel_count1;
          index += cpu_mapping2[i * in_channel_count2 + j2 + 1] - cpu_mapping2[i * in_channel_count2 + j2];
        }
        cpu_mapping_out[i * concat_channel_count + j + 1] = index;
      }
    }
    cudaMemcpy(o_mapping.data(), &cpu_mapping_out[0], o_mapping.dimension(0) * sizeof(int), cudaMemcpyHostToDevice);
  }
  if(data_entry_count1 > 0){
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count1, d);
    concat_values1<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, 
      i_ind1.data(), 
      i_val1.data(), 
      i_sh1.data(), 
      o_sh.data(),
      i_mapping1.data(),
      o_mapping.data(),
      o_ind.data(), 
      o_val.data());
  }
  if(data_entry_count2 > 0){
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count2, d);
    concat_values2<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, 
      i_ind2.data(), 
      i_val2.data(), 
      i_sh2.data(), 
      o_sh.data(), 
      in_channel_count1, 
      i_mapping2.data(),
      o_mapping.data(),
      o_ind.data(), 
      o_val.data());
  }

}


template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConcatBackPropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  const Tensor  *in_indices1, *in_values1, *in_shape1, *in_block_ptr_t1, 
                *in_indices2, *in_values2, *in_shape2, *in_block_ptr_t2, 
                *out_indices, *out_values, *out_shape, *out_block_ptr_t, *gradients;
  OP_REQUIRES_OK(context, context->input("in_indices1", &in_indices1));
  OP_REQUIRES_OK(context, context->input("in_values1", &in_values1));
  OP_REQUIRES_OK(context, context->input("in_shape1", &in_shape1));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping1", &in_block_ptr_t1));
  OP_REQUIRES_OK(context, context->input("in_indices2", &in_indices2));
  OP_REQUIRES_OK(context, context->input("in_values2", &in_values2));
  OP_REQUIRES_OK(context, context->input("in_shape2", &in_shape2));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping2", &in_block_ptr_t2));
  OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
  OP_REQUIRES_OK(context, context->input("out_values", &out_values));
  OP_REQUIRES_OK(context, context->input("out_shape", &out_shape));
  OP_REQUIRES_OK(context, context->input("out_block_channel_mapping", &out_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("grads", &gradients));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh1 = in_shape1->flat<IndiceT>();
  auto i_ind1 = in_indices1->flat<IndiceT>();
  auto i_val1 = in_values1->flat<T>();
  auto i_mapping1 = in_block_ptr_t1->flat<int>();
  auto i_sh2 = in_shape2->flat<IndiceT>();
  auto i_ind2 = in_indices2->flat<IndiceT>();
  auto i_val2 = in_values2->flat<T>();
  auto i_mapping2 = in_block_ptr_t2->flat<int>();
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_mapping = out_block_ptr_t->flat<int>();
  auto grads = gradients->flat<T>(); 
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
    concat_backprop_values<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config, 
      o_ind.data(), 
      i_sh1.data(), 
      i_sh2.data(), 
      o_sh.data(), 
      i_mapping1.data(), 
      i_mapping2.data(), 
      o_mapping.data(), 
      grads.data(), 
      bp1.data(), 
      bp2.data());
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
