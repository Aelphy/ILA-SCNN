#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_conv_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

namespace tensorflow {

//Compress [batch, x, y, ..., channel] indices into a [1D] key while keeping the data sorted.
template <typename dtype>
__global__ void index_KDto1D(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, const int dimension_count, const int entry_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    dtype val = 0;
    dtype mul = 1;
    dtype idx = x;
    for(int i = dimension_count - 1; i >=0; --i) { //exclude channel
      idx = x * dimension_count +  i;
      val = val + mul * in_ptr[idx];
      mul = mul * in_shape_ptr[i];
    }
    out_ind_ptr[x] = val;
  }
}

//TODO: merge device and global function
//Compress [batch, x, y, ..., channel] indices into a [1D] key while keeping the data sorted.
template <typename dtype>
__device__ __forceinline__ void index_KDto1D_(const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, const int dimension_count){
  dtype val = 0;
  dtype mul = 1;
  dtype idx = 0;
  for(int i = dimension_count - 1; i >=0; --i) { //exclude channel
    idx = i;
    val = val + mul * in_ptr[idx];
    mul = mul * in_shape_ptr[i];
  }
  out_ind_ptr[0] = val;
}

//decompress 1D key + channel into K dimensional indices
template <typename dtype>
__device__ __forceinline__ void index_1DtoKD(const int x_in, const int x_out, const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* __restrict__ out_ind_ptr, const int dimension_count){
  dtype idx_out = x_out * dimension_count;
  //1. compressed 1d key, except channel
  dtype *fact = new dtype[dimension_count];
  fact[dimension_count - 1] = 1;
  for(int i = dimension_count - 2; i >= 0; i = i - 1){
    fact[i] = fact[i + 1] * in_shape_ptr[i + 1];
  }
  dtype r = in_index_1d;
  for(int i = 0; i < dimension_count; ++i){
    out_ind_ptr[idx_out + i] = r / fact[i];
    r = r % fact[i];
  }
  delete[] fact;
}


template <typename dtype, typename itype>
__global__ void result_to_output(CudaLaunchConfig config, const itype* __restrict__ index_1d_corr, const itype* __restrict__ in_idx, const itype* __restrict__ out_ind_id, 
    const dtype* __restrict__ res_val, const int index_count, const int data_dimension, itype* __restrict__ index_out, dtype* __restrict__ val_out){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(res_val[x] == 0){
      continue;
    }
    itype idx = x % index_count;
    itype channel = (x - idx) / index_count;
    itype idx2 = index_1d_corr[idx] * data_dimension;
    itype idx3 = (out_ind_id[x] - 1) * data_dimension;
    memcpy(&index_out[idx3], &in_idx[idx2], (data_dimension - 1) * sizeof(itype));
    index_out[idx3 + data_dimension - 1] = channel;
    val_out[out_ind_id[x] - 1] = res_val[x]; 
  }
}

//mark unique elemets in an array with a $1$
template <typename dtype>
__global__ void compute_unique_mask(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, dtype* out_ptr){
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

template <typename dtype>
__global__ void get_array_channel(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, dtype* out_ptr, int channel_id, int data_dimension){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    out_ptr[x] = in_ptr[x * data_dimension + channel_id];
  }
}

//mark non-zero elemets in an array with a $1$
template <typename dtype, typename itype>
__global__ void non_zero_mask(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, itype* __restrict__ out_ptr){
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

//copy obtain unique elemets from array
template <typename dtype, typename itype>
__global__ void unique_array(CudaLaunchConfig config, const dtype* __restrict__ in_id_ptr, const itype* __restrict__ unique_masked_ptr, 
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
template <typename dtype, typename itype>
__global__ void compute_segment_start(CudaLaunchConfig config, itype* data_offset, const dtype* masked_indices, const dtype* unique_count){
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

//copy obtain unique elemets from array
template <typename dtype, typename itype>
__global__ void compute_segment_end(CudaLaunchConfig config, itype* offset, const itype* __restrict__ segment_start, const dtype* __restrict__ count, const int filter_weight_count){
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
template <typename dtype, typename itype>
__global__ void apply_sorted_indices(CudaLaunchConfig config, dtype* sorted, const dtype* __restrict__ unsorted, const itype* __restrict__ corresponds){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) { //x might overflow when testing extreme case
      break;
    }
    sorted[x] = unsorted[corresponds[x]];
  }
}

//exact binary search
template<typename dtype>
__device__ __forceinline__ void index_lookup(const dtype index, const dtype *data, const dtype data_start,  const dtype data_end,
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
template <typename dtype, typename itype>
__global__ void compute_block_start(CudaLaunchConfig config,  const itype* __restrict__ unique_masked_ptr, 
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
template <typename dtype>
__global__ void prepare_filter_weights_(CudaLaunchConfig config, 
                  const dtype* __restrict__ f_id_ptr, const dtype* __restrict__ f_sh_ptr, const dtype* __restrict__ in_sh_ptr,  
                  dtype* out_id_ptr, dtype* out_ch_ptr, dtype* in_channel, dtype* index, const int dimension_count, const int filter_entry_count){
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
    for(int i = dimension_count - 2; i > 0; --i) {
      const int f_i = i - 1;
      const dtype offset = (f_sh_ptr[f_i] - 1)/2;
      idx = x * dimension_count +  f_i;  
      val = val + mul * (offset - f_id_ptr[idx]); //flip filter weights
      mul = mul * in_sh_ptr[i];
    }
    //const dtype channel = in_ptr[(x + 1)  * dimension_count - 1];
    out_id_ptr[x] = val;
    out_ch_ptr[x] = f_id_ptr[x * dimension_count + dimension_count - 1];
    in_channel[x] = f_id_ptr[x * dimension_count + dimension_count - 2];
    index[x] = x;
  }
}

template<typename DeviceT, typename T, typename IndiceT> inline void
preprocess_filter(OpKernelContext* context, DeviceT d, const IndiceT* f_ids_kd, const T* f_vals, const IndiceT* f_shape, const IndiceT* i_shape,  int data_dimension, int filter_weight_count,
    int** filter_segments_start, int** filter_segments_end, T** filter_sorted_weights, IndiceT** filter_sorted_ind_1d, int& filter_segment_count_)
{
  IndiceT *unique_masked = 0;
  checkCuda(cudaMalloc(&unique_masked, filter_weight_count * sizeof(IndiceT)));
  IndiceT *unique_count = 0;
  checkCuda(cudaMalloc(&unique_count, filter_weight_count * sizeof(IndiceT)));
  IndiceT *filter_ind_1d = 0;
  checkCuda(cudaMalloc(&filter_ind_1d, filter_weight_count * sizeof(IndiceT)));
  IndiceT *filter_out_channel = 0;
  checkCuda(cudaMalloc(&filter_out_channel, filter_weight_count * sizeof(IndiceT)));
  IndiceT *filter_in_channel = 0;
  checkCuda(cudaMalloc(&filter_in_channel, filter_weight_count * sizeof(IndiceT)));
  IndiceT *filter_id = 0;
  checkCuda(cudaMalloc(&filter_id, filter_weight_count * sizeof(IndiceT)));
  CudaLaunchConfig config_f1d = GetCudaLaunchConfig(filter_weight_count, d);
  prepare_filter_weights_<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d,
    f_ids_kd, f_shape, i_shape, filter_ind_1d, filter_out_channel, filter_in_channel, filter_id, data_dimension, filter_weight_count);
  cudaDeviceSynchronize();
  //sort filter w.r.t. output and input channels
  IndiceT* new_filter_indice = 0;
  IndiceT* filter_sorted_out = 0;
  IndiceT* filter_sorted_in = 0;
  IndiceT* filter_sorted_tmp_c_in = 0;
  IndiceT filter_segment_count = 0;
  checkCuda(cudaMalloc(&new_filter_indice, filter_weight_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&filter_sorted_out, filter_weight_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&filter_sorted_in, filter_weight_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&filter_sorted_tmp_c_in, filter_weight_count * sizeof(IndiceT)));
  compute_sort(context, d, filter_out_channel, filter_sorted_out, filter_id, new_filter_indice, filter_weight_count);
  compute_unique_mask<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_out, unique_masked);
  compute_scan(context, d, unique_count, unique_masked, filter_weight_count);
  cudaMemcpy(&filter_segment_count, unique_count + filter_weight_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  checkCuda(cudaMalloc(filter_segments_start, filter_weight_count * sizeof(int)));
  checkCuda(cudaMalloc(filter_segments_end, filter_weight_count * sizeof(int)));
  cudaDeviceSynchronize();
  compute_segment_start<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_segments_start, unique_masked, unique_count);
  cudaDeviceSynchronize();
  compute_segment_end<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_segments_end, *filter_segments_start, unique_count, filter_weight_count);
  cudaDeviceSynchronize();
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_tmp_c_in, filter_in_channel, new_filter_indice);
  cudaDeviceSynchronize();
  compute_segmented_sort(context, d, filter_sorted_tmp_c_in, filter_sorted_in, new_filter_indice, filter_id, filter_weight_count, filter_segment_count, *filter_segments_start, *filter_segments_end);
  cudaDeviceSynchronize();
  checkCuda(cudaMalloc(filter_sorted_weights, filter_weight_count * sizeof(T)));
  checkCuda(cudaMalloc(filter_sorted_ind_1d, filter_weight_count * sizeof(IndiceT)));
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_sorted_weights, f_vals, filter_id);
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_sorted_ind_1d, filter_ind_1d, filter_id);
  filter_segment_count_ = filter_segment_count; 
  cudaDeviceSynchronize();
  
  cudaFree(unique_masked);
  cudaFree(unique_count);
  cudaFree(filter_ind_1d);
  cudaFree(filter_id);
  cudaFree(filter_out_channel);
  cudaFree(filter_in_channel);
  cudaFree(filter_sorted_tmp_c_in);
  cudaFree(new_filter_indice);
  cudaFree(filter_sorted_out);
  cudaFree(filter_sorted_in);
}

//Compress [batch, x, y, ...] indices into a [1D] key while voxelization
template <typename dtype>
__global__ void compute_voxel_id1D(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, dtype* out_id_ptr, const int dimension_count, const int entry_count, const int hypercube_size, bool ignore_channel = true){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0 || x >= entry_count) {  //x might overflow when testing extreme case
      break;
    }
    dtype val = 0;
    dtype mul = 1;
    dtype idx = x;
    for(int i = dimension_count - 1; i >=0; --i) { //reorder dimensions to [dim1, ..., dimx, batch, channel] and compress
      int ii = i;
      if(i == dimension_count - 1){
        idx = (x + 1) * dimension_count - 1;
        if(!ignore_channel) val = val + mul * in_ptr[idx];
        mul = mul * in_shape_ptr[dimension_count - 1];
      } else if(i == dimension_count - 2){
        idx = x * dimension_count;
        val = val + mul * in_ptr[idx];
        mul = mul * in_shape_ptr[0];
      } else {
        ii = i + 1;
        idx = x * dimension_count + ii;
        val = val + mul * floor(float(in_ptr[idx]) / hypercube_size);
        mul = mul * in_shape_ptr[ii] / hypercube_size;
      }
    }
    out_ind_ptr[x] = val;
    out_id_ptr[x] = x;
  }
}

//decompress id of compressed sparse blocks (does not revert scaling of [dim1, ..., dimx])
template <typename dtype>
__device__ __forceinline__ void decompress_block_id(const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* __restrict__ out_ind_ptr, const int dimension_count, const int hypercube_size){
  //1. compressed 1d key, except channel
  dtype *fact = new dtype[dimension_count];
  dtype *ids = new dtype[dimension_count]; //reorder dimensions to [dim1, ..., dimx, batch, channel]
  dtype *scale = new dtype[dimension_count]; //reorder dimensions to [dim1, ..., dimx, batch, channel]
  for(int i = 0; i < dimension_count - 2; ++i){ 
    ids[i] = i + 1;
    scale[i] = hypercube_size;
  }
  //TODO: Check order of indices of scale and ids
  ids[dimension_count - 2] = 0;
  scale[dimension_count - 2] = 1;
  ids[dimension_count - 1] = dimension_count - 1;
  scale[dimension_count - 1] = 1;
  fact[dimension_count - 1] = 1;
  for(int i = dimension_count - 2; i >= 0; i = i - 1){
    fact[i] = fact[i + 1] / scale[i + 1] * in_shape_ptr[ids[i + 1]];
  }
  dtype r = in_index_1d;
  for(int i = 0; i < dimension_count; ++i){
    out_ind_ptr[ids[i]] = r / fact[i] * scale[i];
    r = r % fact[i];
  }
  delete[] fact;
  delete[] ids;
  delete[] scale;
}

template<typename DeviceT, typename T, typename IndiceT> inline void
coo_to_blocks(  OpKernelContext* context, DeviceT d, const IndiceT* in_ids_kd, const T* in_vals, const IndiceT* in_shape, IndiceT** block_ids_1d, T** block_vals,
                  IndiceT** block_pointer, IndiceT** block_pointer_ids, int dimension_count, int data_entry_count, int hypercube_size, int& block_count)
{
  std::stringstream dout_s;
  IndiceT *tmp_data = 0, *tmp_data2 = 0, *tmp_data3;
  IndiceT *sorted_voxel_ids = 0, *sorted_id = 0;
  checkCuda(cudaMalloc(&tmp_data, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&tmp_data2, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&tmp_data3, data_entry_count * sizeof(IndiceT)));
  CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
  auto &voxel_id = tmp_data;
  auto &data_id = tmp_data2;
  auto &sorted_id_tmp = tmp_data3;
  compute_voxel_id1D<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, in_shape, voxel_id, data_id, dimension_count, data_entry_count, hypercube_size); 
  cudaDeviceSynchronize();
  //1. put entries per block into consecutive segments of a list
  checkCuda(cudaMalloc(&sorted_voxel_ids, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&sorted_id, data_entry_count * sizeof(IndiceT)));
  compute_sort(context, d, voxel_id, sorted_voxel_ids /*sorted voxel ids*/, data_id, sorted_id_tmp /*sorted data ids*/, data_entry_count);
  cudaDeviceSynchronize();
  //2. set pointers to the start of each block
  auto &unique_mask = tmp_data;
  auto &unique_count = tmp_data2;
  compute_unique_mask<IndiceT><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, sorted_voxel_ids, unique_mask);
  cudaDeviceSynchronize();
  compute_scan(context, d, unique_count, unique_mask, data_entry_count);
  cudaDeviceSynchronize();
  IndiceT block_count_ = 0;
  cudaMemcpy(&block_count_, unique_count + data_entry_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  block_count = block_count_;
  checkCuda(cudaMalloc(block_pointer, (block_count_ + 1) * sizeof(IndiceT)));
  IndiceT dec = data_entry_count;
  cudaMemcpy(&(*block_pointer)[block_count_], &dec, sizeof(IndiceT), cudaMemcpyHostToDevice);
  checkCuda(cudaMalloc(block_pointer_ids, block_count_ * sizeof(IndiceT)));
  compute_block_start<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, unique_mask, unique_count, sorted_voxel_ids, *block_pointer, *block_pointer_ids);
  cudaDeviceSynchronize();
  //3. sort w.r.t. channels within each block
  int *segments_start = 0, *segments_end = 0;
  checkCuda(cudaMalloc(&segments_start, block_count * sizeof(int)));
  checkCuda(cudaMalloc(&segments_end, block_count * sizeof(int)));
  compute_segment_start<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, segments_start, unique_mask, unique_count);
  compute_segment_end<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, segments_end, segments_start, unique_count, data_entry_count);
  auto &channel = tmp_data;
  cudaDeviceSynchronize();
  get_array_channel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, channel, dimension_count - 1, dimension_count);
  cudaDeviceSynchronize();
  compute_segmented_sort(context, d, channel, tmp_data2, sorted_id_tmp, sorted_id, data_entry_count, block_count, segments_start, segments_end);
  cudaDeviceSynchronize();
  //4. apply block structure to data
  auto &id1d = tmp_data;
  index_KDto1D<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, in_shape, id1d, dimension_count, data_entry_count);
  checkCuda(cudaMalloc(block_ids_1d, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(block_vals, data_entry_count * sizeof(T)));
  apply_sorted_indices<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, *block_ids_1d, id1d, sorted_id);
  apply_sorted_indices<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, *block_vals, in_vals, sorted_id);
  //# free temporary resources 
  cudaDeviceSynchronize();
  cudaFree(tmp_data);
  cudaFree(tmp_data2);
  cudaFree(tmp_data3);
  cudaFree(segments_start);
  cudaFree(segments_end);
  cudaFree(sorted_voxel_ids);
  cudaFree(sorted_id);
}

template <typename dtype>
__device__ __forceinline__ void map_index_1d_to_buffer_id(const dtype in_id, const dtype* __restrict__ in_shape_ptr,
                                                          const dtype* out_id, const int hypercube_size)
{
  //TODO
}

template <typename dtype>
__device__ __forceinline__ void map_shared_to_channel_buffer(const dtype in_id, const int hypercube_size,
                                                          const dtype* out_id, const dtype* __restrict__ in_shape_ptr)
{
  //TODO
}

template <typename dtype, typename itype>
__global__ void approxSparseDirectConv(CudaLaunchConfig config,
   const itype* __restrict__ in_block_ids, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_block_pointer, 
   const itype* __restrict__ in_shape_ptr, const int block_start_, 
   const itype* __restrict__ filter_ind_1d, const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape, const int* __restrict__ filter_segments_start, 
   const int* __restrict__ filter_segments_end, const int filter_segment_count, const int filter_start, const int filter_end,
   const itype* __restrict__ out_sh, dtype* dense_channel_buffer,
   int data_entry_count, int filter_weight_count, int data_dimension, int hypercube_size)
{
  __shared__ dtype accumulated_result[3072]; //TODO: dynamic allocation
  const int block_id = blockIdx.x + block_start_; //TODO check block_start_
  const int block_start = in_block_pointer[block_id];
  const int block_end = in_block_pointer[block_id + 1];
  const int filter_weight_start = filter_start;
  const int filter_weights_end = filter_end;
  const int ch_filter_weight_count = filter_weights_end - filter_weight_start;
  const int input_data_count = block_end - block_start;
  const int operation_count = ch_filter_weight_count * input_data_count;

  for(int x = threadIdx.x; x < operation_count; x += blockDim.x){
    //TODO: check indexing of loop
    const int did = x % filter_weight_count;
    const int fid = (x - did) /  filter_weight_count;
    const int block_data_id = did + block_start;
    const int filter_data_id = fid + filter_weight_start;
    const dtype fv = filter_weights[filter_data_id];
    const itype fi = filter_ind_1d[filter_data_id];
    const dtype iv = in_block_vals[block_data_id];
    const itype ii = in_block_ids[block_data_id];
    const itype out_i = fi + ii;
    const dtype out_v = fv * iv;
    itype acc_id;
    map_index_1d_to_buffer_id(out_i, in_shape_ptr, &acc_id, hypercube_size);
    //TODO: map out_i to accumulated_result; decompose out_i to find index in accumulated result
    atomicAdd(&accumulated_result[acc_id], out_v);
  }
  __syncthreads();
  //check if entries are valid (inside tensor shape) and write valid entries to global memory buffer
  //TODO
  for(int x = threadIdx.x; x < pow(hypercube_size, data_dimension - 2); x += blockDim.x){
    itype local_id = x;
    if(accumulated_result[local_id] == 0) continue;
    itype global_acc_id;
    //TODO: check if index is in bounds of out shape
    map_shared_to_channel_buffer(local_id, hypercube_size, &global_acc_id, in_shape_ptr);
    atomicAdd(&dense_channel_buffer[global_acc_id], accumulated_result[local_id]);
  }
}

//generate dense lookup table for blocks in each batch and channel
template <typename dtype>
__global__ void compute_input_block_index(CudaLaunchConfig config, dtype* in_block_id, dtype* in_block_ptr, int* out_index_ptr, const int number_blocks, const int number_batches,
const int number_channels){
  //initialize values to 0
  dtype op_count = number_batches * number_channels;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(x < op_count){
      out_index_ptr[x] = op_count; //not defined
    }

  }
  __syncthreads();
  //find existing correspondences
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(x >= number_blocks) continue;
    //TODO get channel and batch from in_block_ptr[in_block_id[x]]; //each block covers exactly one channel and belongs to one batch!
    int channel = 0; //TODO
    int batch = 0; //TODO
    atomicMin(&out_index_ptr[batch * number_channels + channel], x);
  }
  __syncthreads();
  //fix non existing correspondences
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    //TODO: better parallelization
    if(out_index_ptr[x] == op_count){
      bool fixed = false;
      for(int i = x + 1; i < op_count; ++i){ //linear search to the end until valid entry is found or number_blocks
        if(out_index_ptr[i] != op_count){
          out_index_ptr[x] = out_index_ptr[i];
          fixed = true;
          break;
        }
      }
      if(!fixed){
        out_index_ptr[x] = number_blocks;
      }
    }
  }
}

namespace functor {
template <typename DeviceT, typename T, typename IndiceT>
void ApproxDirectSparseConvFunctor<DeviceT, T, IndiceT>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, const IndiceT& filter_dim) const {
  clock_t t_total = clock();
  const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  const DeviceT d = context->eigen_device<DeviceT>();
  int device_version = d.majorDeviceVersion();
  if(device_version < 6){
    LOG(WARNING) << "compute capability to low; requires 6.0 or higher for fast sparse convolution" << std::endl; //atomics for shared memory, max blocks per sm, etc...
  }
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->matrix<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto f_sh = filter_shape->flat<IndiceT>();
  auto f_ind = filter_indices->matrix<IndiceT>();
  auto f_val = filter_values->flat<T>(); 
  const int data_entry_count = i_ind.dimension(0);
  const int data_dimension = i_ind.dimension(1);
  const int filter_weight_count = f_ind.dimension(0);
  const int smpb = d.sharedMemPerBlock();
  const int mtpb = d.maxCudaThreadsPerBlock();
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  std::vector<IndiceT> cpu_in_shape(data_dimension);
  cudaMemcpy(&cpu_in_shape[0], i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  IndiceT channel_dense_size = 1;
  for(size_t i = 1; i < data_dimension - 2; ++i){
    channel_dense_size = (channel_dense_size * cpu_in_shape[i + 1]); //x,y,z, ... (no channel and no batch) rounded up
  }
  const int batch_count = cpu_in_shape[0];
  const int in_channel_count = cpu_in_shape[data_dimension - 1];

  std::stringstream dout_s;
  //indices must! be sorted
  clock_t t;
  
  //preprocessing step (1) has to be performed only for one layer in the neural network! Also step (2) can be precomputed and shouldn't affect runtime of nn
  
  /////
  //1. Convert Coordinate List Format to sparse block format and compress k dimensional indices to 1d
  t = clock();
  int block_count = 0;
  int filter_size = 3;
  const int hypercube_size = floor(pow(float(smpb / sizeof(T) / 4), 1. / filter_dim)) - (filter_size - 1); //compute block size: assumptions: i) all dimensions have the same size (not necessarly true); ii) two blocks per sm
  if(hypercube_size <= 0) return; //TODO: THROW ERROR
  IndiceT *in_block_ids = 0, *in_block_pointer = 0, *in_block_pointer_ids = 0;
  T *in_block_vals = 0;
  coo_to_blocks(context, d, i_ind.data(), i_val.data(), i_sh.data(), &in_block_ids, &in_block_vals, &in_block_pointer, &in_block_pointer_ids, data_dimension, data_entry_count, hypercube_size, block_count);
  LOG(INFO) << "Edge length: " << hypercube_size << " Shared memory per block: " << smpb << " sizeof T " << sizeof(T) << std::endl;
  dout_s << "t1: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  LOG(INFO) << dout_s.str(); dout_s.str("");

  /////
  //2. prepare filter: directly manipulate 1D keys instead of kD indices and flip filter weights to be applicable for direct convolution and sort filter w.r.t. output and input channels
  t = clock();
  int *filter_segments_start = 0, *filter_segments_end = 0;
  int filter_segment_count_ = 0;
  T* filter_sorted_weights = 0;
  IndiceT* filter_sorted_ind_1d = 0;
  preprocess_filter(context, d, f_ind.data(), f_val.data(), f_sh.data(), i_sh.data(), data_dimension, filter_weight_count, &filter_segments_start, &filter_segments_end, &filter_sorted_weights, &filter_sorted_ind_1d, filter_segment_count_);
  dout_s << "t2: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  LOG(INFO) << dout_s.str(); dout_s.str("");

  /////
  //3. get mapping of blocks / channels (input) and input channels / output channels (filter)
  t = clock();
  dout_s << "t2: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  int* filter_channel_mapping = 0;
  int* input_block_mapping = 0;

  /////
  //4. compute out shape
  //TODO
  IndiceT *out_sh = 0;
  checkCuda(cudaMalloc(&out_sh, data_dimension * sizeof(IndiceT)));
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //5. perform first convolution to know sparse output shape
  t = clock();
  T* channel_buffer = 0;
  checkCuda(cudaMalloc(&channel_buffer, channel_dense_size * sizeof(T))); //allocate dense memory for one channel
  IndiceT conv_out_count = out_channel_count * 1;
  CudaLaunchConfig config_conv = GetCudaLaunchConfig(0, d);
  for(int i = 0; i < batch_count; ++i){
    for(int j = 0; j < out_channel_count; ++j){
      int block_start = 0; //TODO;
      int block_end = 0; //TODO
      cudaDeviceSynchronize();
      cudaMemset(channel_buffer, 0, conv_out_count * sizeof(T));  //stores the dense result of the computed output channel in buffer
      for(int k = 0; k < in_channel_count; ++k){
        int filter_start = 0; //TODO
        int filter_end = 0; //TODO
        config_conv.thread_per_block = 0; //TODO: set number threads and blocks
        config_conv.block_count = block_end - block_start; //TODO
        approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
          in_block_ids, in_block_vals, in_block_pointer, i_sh.data(), block_start,
          filter_sorted_ind_1d, filter_sorted_weights, f_sh.data(), filter_segments_start, filter_segments_end, filter_segment_count_, filter_start, filter_end,
          out_sh, channel_buffer, data_entry_count, filter_weight_count, data_dimension, hypercube_size);
      }
      cudaDeviceSynchronize(); 
      //TODO: Get result for channel; overwrite global dense accumulation buffer with 0 and continue with next channel
    }
  }
  dout_s << "t4: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;
  auto result_count = conv_out_count;
  LOG(INFO) << dout_s.str(); dout_s.str("");

  /////
  //6. Create output tensor and fill it in a second run of convolution
  t = clock();
  Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL;
  if(result_count < 0 || result_count > conv_out_count) result_count = 0; //TODO: debug this case should not happen
  TensorShape out_ind_shape = {(IndiceT) result_count, (IndiceT) data_dimension};
  TensorShape out_val_shape = {(IndiceT) result_count};
  TensorShape out_sh_shape = {(IndiceT) data_dimension};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->matrix<IndiceT>();
  auto o_val = out_values->flat<T>();
  cudaDeviceSynchronize();
  cudaMemcpy(o_sh.data(), out_sh, data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  dout_s << "t7: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  //# free memory
  t = clock();
  cudaFree(filter_segments_start);
  cudaFree(filter_segments_end);
  cudaFree(filter_sorted_ind_1d);
  cudaFree(filter_sorted_weights);
  cudaFree(channel_buffer);
  cudaFree(out_sh);
  cudaFree(in_block_ids);
  cudaFree(in_block_pointer);
  cudaFree(in_block_pointer_ids);
  cudaFree(in_block_vals);
  dout_s << "t8: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  cudaDeviceSynchronize();
  dout_s << "t_total: " << float(clock() - t_total)/CLOCKS_PER_SEC << std::endl;

  LOG(INFO) << dout_s.str();
}
}  // end namespace functor

// Instantiate the GPU implementation for float.

//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type) \
 template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, type, indice_type>;
#define INIT_GPU_ALL(type)    \
  INIT_GPU_TYPE(type, int64); \
  INIT_GPU_TYPE(type, int32);

INIT_GPU_ALL(float);
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
