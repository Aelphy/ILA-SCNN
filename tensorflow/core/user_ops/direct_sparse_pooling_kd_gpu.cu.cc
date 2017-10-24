#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_pooling_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions


#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#define EPS__ 1./65536.

namespace tensorflow {

//Compress [batch, x, y, ...] indices into a [1D] key while voxelization
template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_voxel_id1D__(CudaLaunchConfig config, const dtype* __restrict__ in_ptr_1d, const dtype* __restrict__ in_shape_ptr, 
                    const dtype* __restrict__ out_shape_ptr, dtype* out_ind_ptr, const int* voxel_sizes_)
{
  dtype idx_kd[data_dimension];
  int voxel_sizes[data_dimension];
  for(int i = 0; i < data_dimension; ++i){
    voxel_sizes[i] = voxel_sizes_[i];
  }
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    } 
    index_1DtoKD<dtype, data_dimension>(0, in_ptr_1d[x], in_shape_ptr, &idx_kd[0]);
    for(int i = 0; i < data_dimension; ++i){
      idx_kd[i] = dtype(floor(float(idx_kd[i]) / voxel_sizes[i]));
    }
    dtype val = 0;
    dtype mul = 1;
    dtype idx = x;
    for(int i = data_dimension - 1; i >= 0; --i){ //reorder dimensions to [batch, channel, dim1, ..., dimx] and compress
      int ii = i;
      if(i == 1){ 
        val = val + mul * idx_kd[data_dimension - 1];
        mul = mul * out_shape_ptr[data_dimension - 1]; 
      } else if(i == 0){ 
        val = val + mul * idx_kd[0];
        mul = mul * in_shape_ptr[0];
      } else {
        ii = i - 1;
        val = val + mul * idx_kd[ii]; //round value to first entry of block
        mul = mul * out_shape_ptr[ii];
      }   
    }   
    out_ind_ptr[x] = val;
  }
}


template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_coresponces(CudaLaunchConfig config, const dtype* __restrict__ unique_mask, const dtype* __restrict__ in_offset, dtype* out_cor, int out_count)
{
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    if(x == config.virtual_thread_count - 1){
      out_cor[out_count] = config.virtual_thread_count - 1;
    } else if(unique_mask[x] == 1){
      dtype id = in_offset[x] - 1;
      out_cor[id] = x;
    }
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_max_pooling(CudaLaunchConfig config, const itype* __restrict__ data_cor, const itype* __restrict__ in_offset, const itype* __restrict__ in_ids, const dtype* __restrict__ in_vals, const itype* __restrict__ out_shape, itype* out_ids, dtype* out_vals)
{
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    //find maximum for block
    dtype max = 0;
    itype max_id = 0;
    int up_range =  max(data_cor[x], data_cor[x + 1]);
    for(int i = data_cor[x]; i < up_range; ++i){
      if(in_vals[i] > max){
        max = in_vals[i];
        max_id = in_ids[i];
      }
    }
    //convert block id to data id
    decompress_block_id<itype, data_dimension>(max_id, out_shape, &id_kd[0]);
    itype data_max_id;
    index_KDto1D_<itype, data_dimension>(&id_kd[0], out_shape, &data_max_id);
    //write result
    out_ids[x] = data_max_id;
    out_vals[x] = max;
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_max_pooling_backprop(CudaLaunchConfig config, const itype* __restrict__ data_cor, const itype* __restrict__ in_offset, const itype* __restrict__ in_idx_ids, const dtype* __restrict__ in_vals, const itype* __restrict__ out_shape, const itype* __restrict__ out_ids, const dtype* __restrict__ out_vals, const dtype* __restrict__ grads, dtype* backprops)
{
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    dtype grad = grads[x];
    if(grads[x] == 0) continue;
    dtype max = out_vals[x];
    int up_range =  max(data_cor[x], data_cor[x + 1]);
    for(int i = data_cor[x]; i < up_range; ++i){
      if(in_vals[i] >= max - EPS__ && in_vals[i] <= max + EPS__){
        int in_id = in_idx_ids[i];
        backprops[in_id] = grad;
      }
    }
  }
}

template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_out_mapping(CudaLaunchConfig config, const dtype* __restrict__ in_offset, const int* __restrict__ in_mapping, int* out_mapping, int entry_count)
{
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    if(x == config.virtual_thread_count - 1){
      out_mapping[x] = entry_count;
    } else {
      out_mapping[x] = in_offset[in_mapping[x]] - 1;
    }
  }
}

namespace functor {
  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectSparseMaxPoolingFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride) const {
    const Tensor *in_indices, *in_values, *in_shape, *in_block_channel_mapping;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_channel_mapping));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_sh = in_shape->flat<IndiceT>();
    auto i_ind = in_indices->flat<IndiceT>();
    auto i_val = in_values->flat<T>();
    auto i_mapping = in_block_channel_mapping->flat<int>();
    auto bcount = i_mapping.dimension(0);
    int data_entry_count;
    cudaMemcpy(&data_entry_count, i_mapping.data() + bcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
   
    //TODO: filter (per channel)
    //TODO: no atomic max for floating point values! (suboptimal implementation with radix sort) :/
 
    /////
    //1. allocate temp buffer
    Tensor in_out_map_tensor, in_out_map_ids_sorted_tensor, out_sorted_values_tensor, strides_tensor, offset_tensor, batch_channel_count_tensor; 
    IndiceT *in_out_map_ids = 0, *in_out_map_ids_sorted = 0, *offset = 0;
    int32 *strides_ = 0, *tmp_batch_channel_count;
    T *out_sorted_values = 0;
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    allocate_tensor(context, in_out_map_tensor, &in_out_map_ids, data_entry_count);
    allocate_tensor(context, in_out_map_ids_sorted_tensor, &in_out_map_ids_sorted, data_entry_count);
    allocate_tensor(context, out_sorted_values_tensor, &out_sorted_values, data_entry_count);
    allocate_tensor(context, strides_tensor, &strides_, data_dimension);
    allocate_tensor(context, offset_tensor, &offset, data_entry_count + 1);
    allocate_tensor(context, batch_channel_count_tensor, &tmp_batch_channel_count, bcount);
    cudaMemcpy(strides_, &stride[0], data_dimension * sizeof(int32), cudaMemcpyHostToDevice);
    
    /////
    //2. write output shape
    std::vector<IndiceT> cpu_shape(data_dimension);
    cudaMemcpy(&cpu_shape[0], i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL, *out_block_mapping = NULL;
    TensorShape out_sh_shape = {(IndiceT) data_dimension};
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
    auto o_sh = out_shape->flat<IndiceT>();
    for(size_t i = 0; i < data_dimension; ++i){
      cpu_shape[i] = IndiceT(ceil(float(cpu_shape[i]) / stride[i]));
    } 
    cudaMemcpy(o_sh.data(), &cpu_shape[0], data_dimension * sizeof(IndiceT), cudaMemcpyHostToDevice);

    /////
    //3. compute hypercubes for pooling
    cudaStreamSynchronize(d.stream());
    compute_voxel_id1D__<IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), 
      i_sh.data(), o_sh.data(), in_out_map_ids, strides_);
    cudaStreamSynchronize(d.stream());
    compute_sort(context, d, in_out_map_ids, in_out_map_ids_sorted, i_val.data(), out_sorted_values, data_entry_count);
    cudaStreamSynchronize(d.stream());
    IndiceT *unique_mask = in_out_map_ids;
    compute_unique_mask<IndiceT><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_out_map_ids_sorted, unique_mask);
    cudaStreamSynchronize(d.stream());
    compute_scan(context, d, offset, unique_mask, data_entry_count, true); //inclusive scan
    cudaStreamSynchronize(d.stream());
    IndiceT out_count = 0;
    cudaMemcpy(&out_count, offset + data_entry_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
    Tensor data_cor_tensor;
    IndiceT* data_cor;
    allocate_tensor(context, data_cor_tensor, &data_cor, out_count + 1);
    CudaLaunchConfig config1 = GetCudaLaunchConfig(data_entry_count + 1, d);
    compute_coresponces<IndiceT, data_dimension><<<config1.block_count, config1.thread_per_block, 0, d.stream()>>>(config1, unique_mask, offset, data_cor, out_count);
    cudaStreamSynchronize(d.stream());

    /////
    //4. allocate output tensors 
    TensorShape out_ind_shape = {(IndiceT) out_count};
    TensorShape out_val_shape = {(IndiceT) out_count};
    TensorShape out_block1_shape = {(IndiceT) bcount};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_block_channel_mapping", out_block1_shape, &out_block_mapping));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
    auto o_ind = out_indices->flat<IndiceT>();
    auto o_mapping = out_block_mapping->flat<int>();
    auto o_val = out_values->flat<T>();

    /////
    //5. perform pooling within hypercubes
    cudaStreamSynchronize(d.stream());
    CudaLaunchConfig configo = GetCudaLaunchConfig(out_count, d);
    compute_max_pooling<T, IndiceT, data_dimension><<<configo.block_count, configo.thread_per_block, 0, d.stream()>>>(configo, data_cor, offset, in_out_map_ids_sorted, out_sorted_values, o_sh.data(), o_ind.data(), o_val.data());

    CudaLaunchConfig configb = GetCudaLaunchConfig(bcount, d);
    compute_out_mapping<IndiceT><<<configb.block_count, configb.thread_per_block, 0, d.stream()>>>(configb, offset, i_mapping.data(), o_mapping.data(), out_count);
  }


  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectSparseMaxPoolingBackpropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride) const {
    const Tensor *in_indices, *in_values, *in_shape, *in_block_channel_mapping, *gradients;
    const Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL, *out_block_mapping = NULL;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_channel_mapping));
    OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
    OP_REQUIRES_OK(context, context->input("out_values", &out_values));
    OP_REQUIRES_OK(context, context->input("out_shape", &out_shape));
    OP_REQUIRES_OK(context, context->input("out_block_channel_mapping", &out_block_mapping));
    OP_REQUIRES_OK(context, context->input("grads", &gradients));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_sh = in_shape->flat<IndiceT>();
    auto i_ind = in_indices->flat<IndiceT>();
    auto i_val = in_values->flat<T>();
    auto i_mapping = in_block_channel_mapping->flat<int>();
    auto o_sh = out_shape->flat<IndiceT>();
    auto o_ind = out_indices->flat<IndiceT>();
    auto o_val = out_values->flat<T>();
    auto o_mapping = out_block_mapping->flat<int>();
    auto grads = gradients->flat<T>();
    auto bcount = i_mapping.dimension(0);
    int data_entry_count;
    cudaMemcpy(&data_entry_count, i_mapping.data() + bcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
   
    //TODO: filter (per channel)
    //TODO: no atomic max for floating point values! (suboptimal implementation with radix sort) :/
 
    /////
    //1. allocate temp buffer
    Tensor in_out_map_tensor, in_out_map_ids_sorted_tensor, out_sorted_values_tensor, strides_tensor, offset_tensor, batch_channel_count_tensor, idx_tensor, idx_sorted_tensor; 
    IndiceT *in_out_map_ids = 0, *in_out_map_ids_sorted = 0, *offset = 0, *idx_ids = 0, *idx_ids_sorted = 0;
    int32 *strides_ = 0, *tmp_batch_channel_count;
    T *out_sorted_values = 0;
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    allocate_tensor(context, in_out_map_tensor, &in_out_map_ids, data_entry_count);
    allocate_tensor(context, in_out_map_ids_sorted_tensor, &in_out_map_ids_sorted, data_entry_count);
    allocate_tensor(context, idx_tensor, &idx_ids, data_entry_count);
    allocate_tensor(context, idx_sorted_tensor, &idx_ids_sorted, data_entry_count);
    allocate_tensor(context, out_sorted_values_tensor, &out_sorted_values, data_entry_count);
    allocate_tensor(context, strides_tensor, &strides_, data_dimension);
    allocate_tensor(context, offset_tensor, &offset, data_entry_count + 1);
    allocate_tensor(context, batch_channel_count_tensor, &tmp_batch_channel_count, bcount);
    cudaMemcpy(strides_, &stride[0], data_dimension * sizeof(int32), cudaMemcpyHostToDevice);
    
    /////
    //3. compute hypercubes for pooling
    cudaStreamSynchronize(d.stream());
    compute_voxel_id1D__<IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), 
      i_sh.data(), o_sh.data(), in_out_map_ids, strides_);
    init_index_x<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, idx_ids);
    cudaStreamSynchronize(d.stream());
    compute_sort(context, d, in_out_map_ids, in_out_map_ids_sorted, i_val.data(), out_sorted_values, data_entry_count);
    compute_sort(context, d, in_out_map_ids, in_out_map_ids_sorted, idx_ids, idx_ids_sorted, data_entry_count); //TODO: not nice
    cudaStreamSynchronize(d.stream());
    IndiceT *unique_mask = in_out_map_ids;
    compute_unique_mask<IndiceT><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_out_map_ids_sorted, unique_mask);
    cudaStreamSynchronize(d.stream());
    compute_scan(context, d, offset, unique_mask, data_entry_count, true); //inclusive scan
    cudaStreamSynchronize(d.stream());
    IndiceT out_count = 0;
    cudaMemcpy(&out_count, offset + data_entry_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
    Tensor data_cor_tensor;
    IndiceT* data_cor;
    allocate_tensor(context, data_cor_tensor, &data_cor, out_count + 1);
    CudaLaunchConfig config1 = GetCudaLaunchConfig(data_entry_count + 1, d);
    compute_coresponces<IndiceT, data_dimension><<<config1.block_count, config1.thread_per_block, 0, d.stream()>>>(config1, unique_mask, offset, data_cor, out_count);

    /////
    //4. allocate output tensors 
    Tensor *bp;
    TensorShape out_bp_shape = {(IndiceT) i_val.dimension(0)};
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_bp_shape, &bp));
    auto backprops = bp->flat<T>();
    cudaMemset(backprops.data(), 0, backprops.dimension(0) * sizeof(T));

    /////
    //5. perform pooling within hypercubes
    cudaStreamSynchronize(d.stream());
    CudaLaunchConfig configo = GetCudaLaunchConfig(out_count, d);
    compute_max_pooling_backprop<T, IndiceT, data_dimension><<<configo.block_count, configo.thread_per_block, 0, d.stream()>>>(configo, data_cor, offset, idx_ids_sorted, out_sorted_values, o_sh.data(), o_ind.data(), o_val.data(), grads.data(), backprops.data());

  }
} //end namespace functor


#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseMaxPoolingFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseMaxPoolingBackpropFunctor<GPUDevice, type, indice_type, dim>;
#define INIT_GPU_ALL(type, dim)    \
  INIT_GPU_TYPE(type, int64, dim); \
  INIT_GPU_TYPE(type, int32, dim);
#define INIT_GPU_ALL_(type)    \
  INIT_GPU_ALL(type, 5);

INIT_GPU_ALL_(float);
#undef EPS__
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
#undef INIT_GPU_ALL_
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
