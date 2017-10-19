#pragma once

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#include "external/cub_archive/cub/device/device_segmented_radix_sort.cuh"
#include "external/cub_archive/cub/device/device_radix_sort.cuh"
#include "external/cub_archive/cub/device/device_scan.cuh"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/op_kernel.h"

#define PRIME_NUMBER 1900813
#define MAX_1024_THREADS_PER_BLOCK 1024
#define MAX_256_THREADS_PER_BLOCK 256
#define MIN_8_BLOCKS_PER_MP 8

namespace tensorflow {

inline cudaError_t checkCuda(cudaError_t result)
{ 
    if (result != cudaSuccess) {
      LOG(ERROR) << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
      assert(result == cudaSuccess);
    }
    return result;
}


template<typename Device, typename T, typename V>
bool compute_sort(OpKernelContext* ctx, Device& d, const T* d_in_keys, T* d_out_keys, const  V* d_in_values, V* d_out_values, int num_items){
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_in_keys, d_out_keys, d_in_values, d_out_values, num_items, 0, 8 * sizeof(T), d.stream());

  Tensor temp_storage;
  ctx->allocate_temp(
      DT_INT8, TensorShape({static_cast<T>(temp_storage_bytes)}),
      &temp_storage);

  cub::DeviceRadixSort::SortPairs(temp_storage.flat<int8>().data(), temp_storage_bytes, d_in_keys, d_out_keys, d_in_values, d_out_values, num_items, 0, 8 * sizeof(T), d.stream());
  return true; //TODO: check cub return values
}


template<typename Device, typename T, typename V>
bool compute_segmented_sort(OpKernelContext* ctx, Device& d, const T* d_in_keys, T* d_out_keys, const  V* d_in_values, V* d_out_values, int num_items, int segments_count, const int* segments_start, const int* segments_end){
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, d_in_keys, d_out_keys, d_in_values, d_out_values, num_items, segments_count, segments_start, segments_end, (int) 0, (int) 8 * sizeof(T), d.stream());

  Tensor temp_storage;
  ctx->allocate_temp(
      DT_INT8, TensorShape({static_cast<T>(temp_storage_bytes)}),
      &temp_storage);

  cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.flat<int8>().data(), temp_storage_bytes, d_in_keys, d_out_keys, d_in_values, d_out_values, num_items, segments_count, segments_start, segments_end, 0, 8 * sizeof(T), d.stream());
  return true; //TODO: check cub return values
}

template<typename Device, typename T>
bool compute_scan(OpKernelContext* ctx, Device& d, T* __restrict__ out, const T* __restrict__ in, const int count, const bool inclusive = true){
  size_t temp_storage_bytes = 0;
  if(inclusive){
      cub::DeviceScan::InclusiveSum(/*temp_storage*/ nullptr, temp_storage_bytes,
                             /*d_in*/ in, 
                             /*d_out*/ out,
                             /*num_items*/ count,
                             /*stream*/ d.stream());
  } else {
      cub::DeviceScan::ExclusiveSum(/*temp_storage*/ nullptr, temp_storage_bytes,
                             /*d_in*/ in, 
                             /*d_out*/ out,
                             /*num_items*/ count,
                             /*stream*/ d.stream());
  }

  Tensor temp_storage;
  ctx->allocate_temp(
      DT_INT8, TensorShape({static_cast<T>(temp_storage_bytes)}),
      &temp_storage);
  if(inclusive){
      cub::DeviceScan::InclusiveSum(/*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
                             /*d_in*/ in, 
                             /*d_out*/ out,
                             /*num_items*/ count,
                             /*stream*/ d.stream());
  } else {
      cub::DeviceScan::ExclusiveSum(/*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
                             /*d_in*/ in, 
                             /*d_out*/ out,
                             /*num_items*/ count,
                             /*stream*/ d.stream());
  }

  return true;
}

template<typename T> inline void 
allocate_tensor(OpKernelContext* ctx, Tensor& t, T** data, int count){
  ctx->allocate_temp(DT_INT8, TensorShape({static_cast<int64>(count * sizeof(T))}), &t); 
  *data = (T*) t.flat<int8>().data();
}

template<typename T> void 
debug_out(T* data, int count, std::stringstream& dout_s, std::string name = "dbg"){
  std::vector<T> dbg_v2(count);
  dout_s << name << std::endl;
  cudaMemcpy(&dbg_v2[0], data, dbg_v2.size() * sizeof(T), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dbg_v2.size(); ++i){
    dout_s << dbg_v2[i] << " "; 
  }
  dout_s << std::endl;
}

//Compress [batch, x, y, ..., channel] indices into a [1D] key while keeping the data sorted.
template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
index_KDto1D(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, const int entry_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }    
    dtype val = 0; 
    dtype mul = 1; 
    dtype idx = x; 
    for(int i = data_dimension - 1; i >=0; --i) { //exclude channel
      idx = x * data_dimension +  i;
      val = val + mul * in_ptr[idx];
      mul = mul * in_shape_ptr[i];
    }    
    out_ind_ptr[x] = val; 
  }
}

//TODO: merge device and global function
//Compress [batch, x, y, ..., channel] indices into a [1D] key while keeping the data sorted.
template <typename dtype, int data_dimension> __device__ __forceinline__ void 
index_KDto1D_(const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr){
  dtype val = 0; 
  dtype mul = 1; 
  dtype idx = 0; 
  for(int i = data_dimension - 1; i >=0; --i) { //exclude channel
    idx = i;
    val = val + mul * in_ptr[idx];
    mul = mul * in_shape_ptr[i];
  }
  out_ind_ptr[0] = val;
}

//decompress id of compressed sparse blocks (does not revert scaling of [dim1, ..., dimx])
template <typename dtype, int data_dimension> __device__ __forceinline__ void
decompress_block_id(const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* __restrict__ out_ind_ptr, bool reduced = false){
  //1. compressed 1d key, except channel
  dtype fact[data_dimension];
  dtype ids[data_dimension]; //reorder dimensions to [dim1, ..., dimx, batch, channel]
  for(int i = 2; i <= data_dimension - 1; ++i){
    ids[i] = i -1;
  }
  //TODO: Check order of indices of scale and ids
  ids[0] = 0;
  ids[1] = data_dimension - 1;
  fact[data_dimension - 1] = 1;
  for(int i = data_dimension - 2; i >= 0; i = i - 1){
    fact[i] = fact[i + 1] * in_shape_ptr[ids[i + 1]];
  }
  dtype r = in_index_1d;
  for(int i = 0; i < data_dimension; ++i){
    if(!reduced){
      out_ind_ptr[ids[i]] = r / fact[i];
    } else if(ids[i] > 0 && ids[i] < data_dimension - 1){
      out_ind_ptr[ids[i] - 1] = r / fact[i];
    }
    r = r % fact[i];
  }
}

//decompress 1D key + channel into K dimensional indices
template <typename dtype, int data_dimension> __device__ __forceinline__ void
index_1DtoKD(const int x_out, const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* __restrict__ out_ind_ptr){
  dtype idx_out = x_out * data_dimension;
  //1. compressed 1d key, except channel
  dtype fact[data_dimension];
  fact[data_dimension - 1] = 1;
  for(int i = data_dimension - 2; i >= 0; i = i - 1){
    fact[i] = fact[i + 1] * in_shape_ptr[i + 1];
  }
  dtype r = in_index_1d;
  for(int i = 0; i < data_dimension; ++i){
    out_ind_ptr[idx_out + i] = r / fact[i];
    r = r % fact[i];
  }
}

//TODO: merge device for decompression
//decompress 1D key + channel into K dimensional indices
template <typename dtype, int data_dimension> __device__ __forceinline__ void
index_1DtoKD_reduced(const int x_out, const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* __restrict__ out_ind_ptr, const int offset = 1){
  dtype idx_out = x_out * (data_dimension - 2);
  //1. compressed 1d key, except channel
  dtype fact[data_dimension];
  fact[data_dimension - 1] = 1;
  for(int i = data_dimension - 2; i >= 0; i = i - 1){
    fact[i] = fact[i + 1] * in_shape_ptr[i + 1];
  }
  dtype r = in_index_1d;
  for(int i = 0; i < offset; ++i){
    r = r % fact[i];
  }
  for(int i = offset; i < data_dimension - 2 + offset; ++i){
    auto f = r / fact[i];
    out_ind_ptr[idx_out + i - offset] = f;
    r = r % fact[i];
  }
}

//mark unique elemets in an array with a $1$
template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_unique_mask(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, dtype* out_ptr){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(x == 0){
      out_ptr[x] = 1;
    } else {
      if(in_ptr[x] != in_ptr[x - 1]){
        out_ptr[x] = 1;
      } else {
        out_ptr[x] = 0;
      }
    }
  }
}

template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
get_array_channel(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, dtype* out_ptr, int channel_id){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    out_ptr[x] = in_ptr[x * data_dimension + channel_id];
  }
}

//mark non-zero elemets in an array with a $1$
template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
non_zero_mask(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, itype* __restrict__ out_ptr){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(in_ptr[x] != 0){
      out_ptr[x] = 1;
    } else {
      out_ptr[x] = 0;
    }
  }
}

//mark unique elemets in array
template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
unique_array(CudaLaunchConfig config, const dtype* __restrict__ in_id_ptr, const itype* __restrict__ unique_masked_ptr,
              const itype* __restrict__ unique_count, dtype* unique_ptr, dtype*  unique_cor){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(unique_masked_ptr[x] == 1){
      unique_ptr[unique_count[x] - 1] = in_id_ptr[x];
      unique_cor[unique_count[x] - 1] = x;
    }
  }
}

//copy obtain unique elemets from array
template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_segment_start(CudaLaunchConfig config, itype* data_offset, const dtype* masked_indices, const dtype* unique_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) { //x might overflow when testing extreme case
      break;
    }
    if(masked_indices[x] > 0){
      int oid = unique_count[x] - 1;
      data_offset[oid] = x;
    }
  }
}

//obtain start of segments
template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_segment_end(CudaLaunchConfig config, itype* offset, const itype* __restrict__ segment_start, const dtype* __restrict__ count, const int filter_weight_count){
  auto max_size = count[filter_weight_count - 1] - 1;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) { //x might overflow when testing extreme case
      break;
    }
    if(x == max_size){
      offset[x] = filter_weight_count;
    } else if(x < max_size){
      offset[x] = segment_start[x + 1];
    }
  }
}

//apply sorting
template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
apply_sorted_indices(CudaLaunchConfig config, dtype* sorted, const dtype* __restrict__ unsorted, const itype* __restrict__ corresponds){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) { //x might overflow when testing extreme case
      break;
    }
    sorted[x] = unsorted[corresponds[x]];
  }
}

//exact binary search
template<typename dtype> __device__ __forceinline__ void
index_lookup(const dtype index, const dtype *data, const dtype data_start,  const dtype data_end,
dtype* result_id, dtype* lower_limit = NULL, dtype* upper_limit = NULL){
  //binary search
  dtype upper = data_end;
  dtype lower = data_start;
  while(lower <= upper){
    dtype center = (upper + lower) / 2;
    if(data[center] == index){
      *result_id = center;
      return;
    }
    if(index > data[center]){
      lower = center + 1;
    } else {
      upper = center - 1;
    }
  }
  if(lower_limit) *lower_limit = upper;
  if(upper_limit) *upper_limit = lower;
  *result_id = -1;
}

//copy obtain unique elemets from array
template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_block_start(CudaLaunchConfig config,  const itype* __restrict__ unique_masked_ptr,
              const itype* __restrict__ unique_count, const itype* __restrict__ block_value,
              dtype*  unique_cor, dtype* pointer_value){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(unique_masked_ptr[x] == 1){
      unique_cor[unique_count[x] - 1] = x;
      pointer_value[unique_count[x] - 1] = block_value[x];
    }
  }
}

//prepare filter weights
template <typename dtype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
prepare_filter_weights_(CudaLaunchConfig config,
                  const dtype* __restrict__ f_id_ptr, const dtype* __restrict__ f_sh_ptr, const dtype* __restrict__ in_sh_ptr,
                  dtype* out_id_ptr, dtype* out_ch_ptr, dtype* in_channel, dtype* index, const int filter_entry_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    //index type (dtype) must! be signed
    dtype val = 0;
    dtype mul = 1;
    dtype idx = x;
    //data format: [batch, depth, height, width, in_channels]
    //filter format: [depth, height, width, in_channels, out_channels]
    //manipulate depth, height width only and store in and out channels 
    
    index_KDto1D_<dtype, data_dimension>(&f_id_ptr[x * data_dimension], f_sh_ptr, &val);
    //const dtype channel = in_ptr[(x + 1)  * data_dimension - 1];
    out_id_ptr[x] = val;
    out_ch_ptr[x] = f_id_ptr[x * data_dimension + data_dimension - 1];
    in_channel[x] = f_id_ptr[x * data_dimension + data_dimension - 2];
    index[x] = x;
  }
}

//TODO: check sort input data correctly (1. batch, 2. channel, 3. position)
//generate dense lookup table for blocks in each batch and channel
template <typename dtype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_input_block_index(CudaLaunchConfig config, const dtype* in_block_ptr, const dtype* in_block_ids, int* out_index_ptr, const dtype* in_shape_ptr, int number_blocks, int number_batches, int number_channels, int data_entry_count){
  //initialize values to 0
  dtype idKD[data_dimension];
  dtype op_count = number_batches * number_channels;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0 || x > op_count) {  //x might overflow when testing extreme case
      break;
    }
    out_index_ptr[x] = data_entry_count; //not defined
  }
  __syncthreads();
  //find existing correspondences
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(x >= number_blocks) continue;
    index_1DtoKD<dtype, data_dimension>(0, in_block_ids[in_block_ptr[x]], in_shape_ptr, &idKD[0]);
    //index_1DtoKD<dtype, data_dimension>(0, in_block_id[in_block_ptr[x]], in_shape_ptr, idKD);
    int channel = idKD[data_dimension - 1];
    int batch = idKD[0];
    atomicMin(&(out_index_ptr[batch * number_channels + channel]), in_block_ptr[x]);
  }
  __syncthreads();
  //fix non existing correspondences
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0 || x >= op_count){  //x might overflow when testing extreme case
      break;
    }
    //TODO: better parallelization
    if(out_index_ptr[x] == number_blocks){
      for(int i = x + 1; i <= op_count; ++i){ //linear search to the end until valid entry is found or number_blocks
        if(out_index_ptr[i] != number_blocks){
          out_index_ptr[x] = out_index_ptr[i];
          break;
        }
      }
    }
  }
}

//generate dense lookup table for channels in each batch and channel
template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_filter_channel_index(CudaLaunchConfig config, dtype* filter_in_ch, dtype* filter_out_ch, int* out_index_ptr,
      const int in_channel_count, const int out_channel_count, const int filter_weight_count)
{
  int ch_dim = in_channel_count * out_channel_count;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if(x < 0 || x > ch_dim){  //x might overflow when testing extreme case
      break;
    }
    if(x < ch_dim){
      out_index_ptr[x] = -1; //initialize
    } else if(x == ch_dim){
      out_index_ptr[x] = filter_weight_count;
    }
  }
  __syncthreads();
  //find existing correspondences
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(x >= filter_weight_count) break;
    if(x == 0 || filter_in_ch[x] != filter_in_ch[x - 1] || filter_out_ch[x] != filter_out_ch[x - 1]){
      int in_channel = filter_in_ch[x];
      int out_channel = filter_out_ch[x];
      out_index_ptr[out_channel * in_channel_count + in_channel] = x;
    }
  }
  __syncthreads();
  //fix non existing correspondences
  CUDA_1D_KERNEL_LOOP(x, ch_dim) {
    if (x < 0 || x >= ch_dim) {  //x might overflow when testing extreme case
      break;
    }
    if(out_index_ptr[x] == -1){
      for(int i = x + 1; i <= ch_dim; ++i){ //linear search to the end until valid entry is found or number_blocks
        if(out_index_ptr[i] != -1){
          out_index_ptr[x] = out_index_ptr[i];
          break;
        }
      }
    }
  }
}

//Compress [batch, x, y, ...] indices into a [1D] key while voxelization
template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_voxel_id1D(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, dtype* out_id_ptr, const int entry_count, const int hypercube_size, bool ignore_channel = true){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0 || x >= entry_count){  //x might overflow when testing extreme case
      break;
    }
    dtype val = 0;
    dtype mul = 1;
    dtype idx = x;
    for(int i = data_dimension - 1; i >= 0; --i){ //reorder dimensions to [batch, channel, dim1, ..., dimx] and compress
      int ii = i;
      if(i == 1){
        idx = (x + 1) * data_dimension - 1;
        if(!ignore_channel) val = val + mul * in_ptr[idx];
        mul = mul * in_shape_ptr[data_dimension - 1];
      } else if(i == 0){
        idx = x * data_dimension;
        val = val + mul * in_ptr[idx];
        mul = mul * in_shape_ptr[0];
      } else {
        ii = i - 1;
        idx = x * data_dimension + ii;
        val = val + mul * dtype(floor(float(in_ptr[idx]) / hypercube_size)) * hypercube_size; //round value to first entry of block
        mul = mul * in_shape_ptr[ii];
      }
    }
    out_ind_ptr[x] = val;
    out_id_ptr[x] = x;
  }
}



//Hash Table
struct HashConfig {
  int c0_0;
  int c0_i;
  int c1_0;
  int c1_i;
  int c2_0;
  int c2_i;
  int c3_0;
  int c3_i;
  int bucket_count;
  int bucket_size;
  int cuckoo_max_iterations;
};

template <typename dtype>
__device__ __forceinline__ void hash_function(dtype* out_val, const dtype* in_val, const int c_0, const int c_i, const int mod_size){
  *out_val = ((c_0 + c_i * *in_val) % PRIME_NUMBER) % mod_size;
}

template <typename dtype>
__global__ void precompute_bucket_count(CudaLaunchConfig config, 
                  dtype* bucket_count, dtype* bucket_offset, dtype* bucket_id, const dtype* in_values, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    dtype h(0);
    hash_function(&h, &(in_values[x]), hc.c0_0, hc.c0_i, hc.bucket_count);
    bucket_offset[x] = CAtomicAdd(&bucket_count[h], (dtype) 1);
    bucket_id[x] = h;
  }
}

template <typename dtype>
__global__ void check_bucket_count(CudaLaunchConfig config, 
                  dtype* result, const dtype* bucket_count, const float max_bucket_density, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(bucket_count[x] >= hc.bucket_size * max_bucket_density){
      *result = 1;
    }
  }
}

template <typename dtype>
__global__ void init_buckets(CudaLaunchConfig config, 
                  dtype* hash_table, const dtype* h_ids, const dtype* h_offsets, const dtype* in_ids, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    dtype bid = h_ids[x];
    dtype boff = h_offsets[x];
    hash_table[bid * hc.bucket_size + boff] = in_ids[x] + 1;
  }
}

template <typename dtype, typename itype>
__device__ __forceinline__ void querry_hash_value(dtype *outcome, const itype* hash_table, const dtype* hash_values, const itype* querry, const HashConfig hc){
  itype id;
  querry_hash_table(&id, hash_table, querry, hc);
  if(id < 0){
    *outcome = -1;
  } else {
    *outcome = hash_values[id];
  }
}

template <typename dtype>
__device__ __forceinline__ void querry_hash_table(dtype *outcome, const dtype* hash_table, const dtype* querry, const HashConfig hc){
  dtype h(0), h1(0), h2(0), h3(0);
  const int invalid_result = -1;
  const dtype val = *querry;
  hash_function(&h, &val, hc.c0_0, hc.c0_i, hc.bucket_count);
  const int offset = h * hc.bucket_size;

  const int mod = floorf(hc.bucket_size / 3.);
	hash_function(&h1, &val, hc.c1_0, hc.c1_i, mod);
  if(hash_table[h1 + offset] == *querry){
    *outcome = h1 + offset;
    return;
  }
	hash_function(&h2, &val, hc.c2_0, hc.c2_i, mod);
  if(hash_table[h2 + offset + mod] == *querry){
    *outcome = h2 + offset + mod;
    return;
  }
	hash_function(&h3, &val, hc.c3_0, hc.c3_i, mod);
  if(hash_table[h3 + offset + 2 * mod] == *querry){
    *outcome = h3 + offset + 2 * mod;
    return;
  }
  *outcome = invalid_result;
}

template <typename dtype, typename itype>
__global__ void fill_values(CudaLaunchConfig config, 
                  dtype* hash_values, const itype* hash_table, const itype* in_keys, const dtype* in_values, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype hid(-1);
    querry_hash_table(&hid, hash_table, &in_keys[x], hc);
    if(hid >= 0){
      hash_values[hid] = in_values[x];
    }
  }
}

template <typename dtype, typename itype>
__global__ void check_hash_table(CudaLaunchConfig config, 
                  itype* result, const itype* hash_table, const dtype* hash_values, const itype* in_keys, const dtype* in_values, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype hid(-1);
    querry_hash_table(&hid, hash_table, &in_keys[x], hc);
    if(hash_values[hid] != in_values[x]){
      *result = 1;
    }
  }
}

template <typename dtype, typename itype>
__global__ void fill_values2(CudaLaunchConfig config, 
                  dtype* hash_values, const itype* hash_table, const itype* in_keys, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype hid(-1);
    querry_hash_table(&hid, hash_table, &in_keys[x], hc);
    if(hid >= 0){
      hash_values[hid] = x;
    }
  }
}

template <typename dtype, typename itype>
__global__ void check_hash_table2(CudaLaunchConfig config, 
                  itype* result, const itype* hash_table, const dtype* hash_values, const itype* in_keys, const HashConfig hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype hid(-1);
    querry_hash_table(&hid, hash_table, &in_keys[x], hc);
    if(hash_values[hid] != x){
      *result = 1;
    }
  }
}

//mixed rehashing
template <typename dtype>
__global__ void rehash_buckets(CudaLaunchConfig config, 
                  dtype *result, dtype* hash_table, const HashConfig hc){
  dtype bidx = blockIdx.x;
  dtype tidx = threadIdx.x;
  //initialize shared memory
  __shared__ dtype s[1025];
  auto &is_good = s[hc.bucket_size];
  is_good = 1;

  //try to insert items into shared memory:
  int didx = -1;
  dtype data_idx = -1;
  dtype h1(-1), h2(-1), h3(-1);
  const int invalid_subtable = -1;
  int offset = 0;
  if(bidx < hc.bucket_count && tidx < hc.bucket_size){
    s[tidx] = -1;
    offset = bidx * hc.bucket_size;
    didx = offset + tidx;
    data_idx = hash_table[didx] - 1;
    if(didx == 0){
      *result = 0;
    }
  }
  if(data_idx >= 0){ //data exists at entry
    const int mod = floorf(hc.bucket_size / 3.);
    hash_function(&h1, &data_idx, hc.c1_0, hc.c1_i, mod);
    hash_function(&h2, &data_idx, hc.c2_0, hc.c2_i, mod);
    h2 = h2 + mod;
    hash_function(&h3, &data_idx, hc.c3_0, hc.c3_i, mod);
    h3 = h3 + 2 * mod;
  }
  int in_subtable = invalid_subtable;
  __syncthreads();
  for(int i = 0; i < hc.cuckoo_max_iterations; ++i){
    if(data_idx >= 0){ //data exists at entry
      is_good = 1;
      if(in_subtable == invalid_subtable){
        s[h1] = data_idx;
        in_subtable = 0;
      }
    }
    __syncthreads();
    if(data_idx >= 0){ //data exists at entry
      if(in_subtable == 0 && s[h1] != data_idx){
        s[h2] = data_idx;
        in_subtable = 1;
      }
    }
    __syncthreads();
    if(data_idx >= 0){ //data exists at entry
      if(in_subtable == 1 && s[h2] != data_idx){
        s[h3] = data_idx;
        in_subtable = 2;
      }
    }
    __syncthreads();
    if(data_idx >= 0){ //data exists at entry
      if((in_subtable == 2 && s[h3] != data_idx) ||
      (in_subtable == 1 && s[h2] != data_idx) ||
      (in_subtable == 0 && s[h1] != data_idx)){
        is_good = 0;
        in_subtable = invalid_subtable;
      }
    }
    
    __syncthreads();
    if(is_good == 1){
      break;
    }
    __syncthreads();
  }
  __syncthreads();
  if(didx >= 0){
    hash_table[didx] = s[tidx];
    if(is_good == 0){
      *result = 1;
      is_good = 1;
    }
  }
  __syncthreads();
}

//not minimal yet, is minimal needed?
template <typename Device, typename dtype, typename itype>
int initialize_table(OpKernelContext* ctx, Device d, itype** hash_table_, dtype** hash_values_, const itype* in_keys, const dtype* in_vals, 
    const itype data_count, HashConfig& hc, const itype bucket_size = 1024)
{
  if(d.sharedMemPerBlock() / std::max(sizeof(dtype), sizeof(itype)) / 2 < bucket_size) return -1; //error, not enough shared memory, select smaller number of keys
  if(bucket_size > d.maxCudaThreadsPerBlock()) return -2; 

  const float average_bucket_density = 0.7;
  const float max_bucket_density = 1 - (1 - average_bucket_density) / 2;
  hc.bucket_count =  ceil(data_count / (0.7 * bucket_size)); //on average 70% filled buckets
  hc.bucket_size = bucket_size;
  hc.cuckoo_max_iterations = 100;
  itype *bucket_count = 0;
  checkCuda(cudaMalloc(&bucket_count, hc.bucket_count * sizeof(itype)));
  itype *bucket_offset = 0;
  checkCuda(cudaMalloc(&bucket_offset, data_count * sizeof(itype)));
  itype *bucket_id = 0;
  checkCuda(cudaMalloc(&bucket_id, data_count * sizeof(itype)));
  itype *bucket_start = 0;
  checkCuda(cudaMalloc(&bucket_start, hc.bucket_count * sizeof(itype)));
  itype *kernel_result = 0;
  checkCuda(cudaMalloc(&kernel_result, sizeof(itype)));
  itype *hash_table = 0;
  checkCuda(cudaMalloc(hash_table_, hc.bucket_count * hc.bucket_size * sizeof(itype)));
  hash_table = *hash_table_;
  dtype *hash_values = 0;
  checkCuda(cudaMalloc(hash_values_, hc.bucket_count * hc.bucket_size * sizeof(dtype)));
  hash_values = *hash_values_;

  CudaLaunchConfig cfg = GetCudaLaunchConfig(data_count, d);
  CudaLaunchConfig cfg2 = GetCudaLaunchConfig(hc.bucket_count, d);
  CudaLaunchConfig cfg3;
  cfg3.virtual_thread_count = hc.bucket_size * hc.bucket_count;
  cfg3.thread_per_block = hc.bucket_size;
  cfg3.block_count = hc.bucket_count;
  
  std::stringstream dout_s;
  srand (time(NULL));

  itype result_kernel = 1;
  while(result_kernel == 1){
    cudaMemset(hash_table, 0, hc.bucket_count * hc.bucket_size * sizeof(itype));
    cudaMemset(hash_values, 0, hc.bucket_count * hc.bucket_size * sizeof(dtype));
    cudaMemset(kernel_result, 0, sizeof(itype));
    cudaMemset(bucket_count, 0, hc.bucket_count * sizeof(itype));
    hc.c0_0 = rand() ^ 0xffff;
    hc.c0_i = rand() ^ 0xcba9;
    cudaMemset(kernel_result, 0, sizeof(itype));
    precompute_bucket_count<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, bucket_count, bucket_offset, bucket_id, in_keys, hc);
    cudaDeviceSynchronize();
    check_bucket_count<<<cfg2.block_count, cfg2.thread_per_block, 0, d.stream()>>>(cfg2, kernel_result, bucket_count, max_bucket_density, hc);
    cudaMemcpy(&result_kernel, kernel_result, sizeof(itype), cudaMemcpyDeviceToHost);
    if(result_kernel == 1) continue;
  	//compute_scan(ctx, d, bucket_start, bucket_count, hc.bucket_count, false);

    hc.c1_0 = rand() ^ 0xffff;
    hc.c1_i = rand() ^ 0xcba9;
    hc.c2_0 = rand() ^ 0x7531;
    hc.c2_i = rand() ^ 0xbeef;
    hc.c3_0 = rand() ^ 0xd9f1;
    hc.c3_i = rand() ^ 0x337a;
    init_buckets<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, hash_table, bucket_id, bucket_offset, in_keys, hc);
    cudaMemset(kernel_result, 0, sizeof(itype));
    rehash_buckets<<<cfg3.block_count, cfg3.thread_per_block, 0, d.stream()>>>(cfg3, kernel_result, hash_table, hc);
    cudaMemcpy(&result_kernel, kernel_result, sizeof(itype), cudaMemcpyDeviceToHost);
  }

  //fill hash table with values
  cudaDeviceSynchronize();
  fill_values2<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, hash_values, hash_table, in_keys, hc);
  cudaDeviceSynchronize();
  check_hash_table2<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, kernel_result, hash_table, hash_values, in_keys, hc);
  cudaMemcpy(&result_kernel, kernel_result, sizeof(itype), cudaMemcpyDeviceToHost);
  LOG(INFO) << "result: " << result_kernel << std::endl;
  
  cudaFree(bucket_count);
  cudaFree(bucket_offset);
  cudaFree(bucket_id);
  cudaFree(bucket_start);
  cudaFree(kernel_result);
  //cudaFree(hash_table);
  //cudaFree(hash_values);
  return 0;
}

} //namespace tensorflow

#undef PRIME_NUMBER
