#pragma once

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#include "external/cub_archive/cub/device/device_scan.cuh"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/op_kernel.h"

#define PRIME_NUMBER 1900813

namespace tensorflow {

inline
cudaError_t checkCuda(cudaError_t result)
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
  typedef long long int int64_cu;
  uint64_cu* address_as_ull = (uint64_cu*)address;
  int64_cu old = (int64_cu) *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, (uint64_cu) val + assumed);
    } while (assumed != old);
    return old;
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

template<typename itype>
struct HashConfig {
  itype c0_0;
  itype c0_i;
  itype c1_0;
  itype c1_i;
  itype c2_0;
  itype c2_i;
  itype c3_0;
  itype c3_i;
  itype mod_size;
  itype bucket_count;
  itype bucket_size;
};


template <typename dtype>
__device__ __forceinline__ void hash_function(dtype* out_val, const dtype* in_val, const dtype c_0, const dtype c_i, const dtype mod_size){
  *out_val = ((c_0 + c_i * *in_val) % PRIME_NUMBER) % mod_size;
}

template <typename dtype, typename itype>
__global__ void precompute_bucket_count(CudaLaunchConfig config, 
                  dtype* bucket_count, dtype* bucket_offset, const dtype* in_values, const HashConfig<itype> hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    dtype h(0);
    hash_function(&h, &(in_values[x]), hc.c0_0, hc.c0_i, hc.mod_size);
    bucket_offset[x] = CAtomicAdd(&bucket_count[h], (dtype) 1);
  }
}

template <typename dtype, typename itype>
__global__ void check_bucket_count(CudaLaunchConfig config, 
                  dtype* result, const dtype* bucket_count, const HashConfig<itype> hc){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(bucket_count[x] >= hc.bucket_size){
      *result = 1;
    }
  }
}

template <typename Device, typename dtype, typename itype>
int initialize_table(OpKernelContext* ctx, Device d, const itype* in_keys, const dtype* in_vals, 
    const itype data_count, HashConfig<itype>& hc, const itype bucket_size = 512)
{
  if(d.sharedMemPerBlock() / std::max(sizeof(dtype), sizeof(itype)) / 2 < bucket_size) return -1; //error, not enough shared memory, select smaller number of keys
  if(bucket_size > d.maxCudaThreadsPerBlock()) return -2; 

  hc.bucket_count =  ceil(data_count / bucket_size / 0.8); //on average 80% filled buckets
  hc.bucket_size = bucket_size;
  itype *bucket_count = 0;
  checkCuda(cudaMalloc(&bucket_count, hc.bucket_count * sizeof(itype)));
  itype *bucket_offset = 0;
  checkCuda(cudaMalloc(&bucket_offset, data_count * sizeof(itype)));
  itype *bucket_start = 0;
  checkCuda(cudaMalloc(&bucket_count, hc.bucket_count * sizeof(itype)));
  cudaMemset(bucket_start, 0, hc.bucket_count * sizeof(itype));
  itype *kernel_result = 0;
  checkCuda(cudaMalloc(&kernel_result, sizeof(itype)));

  int rand_range0 = pow(2, sizeof(itype) / 4);
  int rand_range1 = pow(2, sizeof(itype) / 8);
  CudaLaunchConfig cfg = GetCudaLaunchConfig(data_count, d);
  CudaLaunchConfig cfg2 = GetCudaLaunchConfig(hc.bucket_count, d);
  itype result_kernel = 1;
  while(result_kernel == 1){
    cudaMemset(kernel_result, 0, sizeof(itype));
    hc.c0_0 = rand() % rand_range0;
    hc.c0_i = rand() % rand_range1 + 1;
    cudaMemset(kernel_result, 0, sizeof(itype));
    precompute_bucket_count<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, bucket_count, bucket_offset, in_keys, hc);
    cudaDeviceSynchronize();
    check_bucket_count<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(cfg, kernel_result, bucket_count, hc);
    cudaMemcpy(&result_kernel, kernel_result, sizeof(itype), cudaMemcpyDeviceToHost);
  }
  compute_scan(ctx, d, bucket_start, bucket_count, hc.bucket_count, false);

  /*CudaLaunchConfig cfg3;
  cfg3.virtual_thread_count = data_count;
  cfg3.thread_per_block = bucket_size;
  itype physical_thread_count = std::min<itype>(bucket_size, cfg.virtual_thread_count);
  cfg3.block_count = std::min(DIV_UP(physical_thread_count, cfg.thread_per_block), d.getNumCudaMultiProcessors());
  */
  result_kernel = 1;
  while(result_kernel == 1){
    hc.c1_0 = rand() % rand_range0;
    hc.c1_i = rand() % rand_range1 + 1;
    hc.c2_0 = rand() % rand_range0;
    hc.c2_i = rand() % rand_range1 + 1;
    hc.c3_0 = rand() % rand_range0;
    hc.c3_i = rand() % rand_range1 + 1;
    cudaMemset(kernel_result, 0, sizeof(itype));
    cudaMemcpy(&result_kernel, kernel_result, sizeof(itype), cudaMemcpyDeviceToHost);
  }
  return 0;
}

} //namespace tensorflow

#undef PRIME_NUMBER
