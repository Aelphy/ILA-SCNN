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

namespace tensorflow {

inline cudaError_t checkCuda(cudaError_t result)
{ 
    if (result != cudaSuccess) {
      LOG(ERROR) << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
      assert(result == cudaSuccess);
    }
    return result;
}


template<typename dtype>
__device__ dtype CAtomicAdd(dtype* address, dtype val){
	return atomicAdd(address, val);
}

__device__ double CAtomicAdd(int64* address, int64 val)
{
  typedef unsigned long long int uint64_cu;
  uint64_cu* address_ = (uint64_cu*) address;
  uint64_cu val_ = (uint64_cu) val;
  return atomicAdd(address_, val_);
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

//cuckoo rehashing
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
