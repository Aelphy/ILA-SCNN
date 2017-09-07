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
__device__ void index_KDto1D_(const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
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
__device__ void index_1DtoKD(const int x_in, const int x_out, const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
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
        mul = mul * in_shape_ptr[ii];
      }
    }
    out_ind_ptr[x] = val;
    out_id_ptr[x] = x;
  }
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

/*template <typename dtype, typename itype>
__global__ void approxSparseDirectConv(CudaLaunchConfig config, 
   const itype* __restrict__ i_ind, const dtype* __restrict__ i_val, const itype* __restrict__ i_sh, const itype* __restrict__ i_ind_1d, const itype* __restrict__ i_ch, //input tensors
   const itype* __restrict__ f_ind, const dtype* __restrict__ f_val, const itype* __restrict__ f_sh, const itype* __restrict__ f_ind_1d, const itype* __restrict__ f_id, //filter tensors
   const itype* __restrict__ r_ind, const itype reduced_count, //search structure for binary search
   dtype* __restrict__ out_conv_data, const itype* __restrict__ out_sh,
   const int data_entry_count, const int filter_weight_count, const int data_dimension){

  //compute data
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) { //index of feature map
     if(x < 0){
       break;
     }
     if(x >= data_entry_count){
       break;
     }
    for(itype y = 0; y < filter_weight_count; ++y){
      //1.a: check channel filter/input
      itype idy = data_dimension * f_id[y];
      //itype idy = data_dimension * y;
      if(i_ch[x] != f_ind[idy + data_dimension - 2]) continue; 
      //1.b check valid indice and valid stride / padding:
      bool is_valid = true;
      itype idx = data_dimension * x;
      for(int i = data_dimension - 2; i > 0; --i){
        itype id = i_ind[idx + i] - f_ind[idy + i - 1] + (f_sh[i - 1] - 1) / 2;
        if(id < 0 || id >= out_sh[i]){
          is_valid = false;
          break;
        }
        //TODO: stride and padding
      }
      if(!is_valid){
        continue;
      }
      //2. compute update indice
      itype lookup_id = i_ind_1d[x] + f_ind_1d[y];
      itype update_id = 0;
      index_lookup(lookup_id, r_ind, (itype) 0, (itype) reduced_count - 1, &update_id); //Binary search
      //querry_hash_value(&update_id, hash_table, hash_values, &lookup_id, hc); //search in hash table
      //3. update indice
      if(update_id < 0){
        continue;
      }
      itype channel_offset = reduced_count * f_ind[idy + data_dimension - 1];
      const float update_val = f_val[y] * i_val[x];
      atomicAdd(&out_conv_data[update_id + channel_offset], update_val);
    }
  }
}
*/


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
  checkCuda(cudaMalloc(block_pointer, block_count_ * sizeof(IndiceT)));
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

//decompress id of compressed sparse blocks (does not revert scaling of [dim1, ..., dimx])
template <typename dtype>
__device__ void decompress_block_id(const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
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
    fact[i] = fact[i + 1] * in_shape_ptr[ids[i + 1]];
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

//copy obtain unique elemets from array
template <typename dtype>
__global__ void compute_dense_block_id(CudaLaunchConfig config,  const dtype* sparse_block_id, dtype* dense_block_id){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    dense_block_id[sparse_block_id[x]] = x + 1; // + 1 to avoid ambiguities with index 0 and non existing blocks
  }
}

template<typename DeviceT, typename T, typename IndiceT> inline void
compute_dense_blocks(DeviceT d, const IndiceT* in_shape, const IndiceT* block_pointer, const IndiceT* block_pointer_ids, IndiceT** dense_block_corr, 
  const int dimension_count, const int data_entry_count, const int hypercube_size, const int block_count)
{
  std::vector<IndiceT> cpu_shape(dimension_count);
  cudaMemcpy(&cpu_shape, in_shape, dimension_count * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  int dim = 1;
  for(size_t i = 0; i < dimension_count - 1; ++i){
    dim = (dim * cpu_shape[i + 1] + hypercube_size - 1) / hypercube_size; //batch, x,y,z, ... (no channel) rounded up
  }
  checkCuda(cudaMalloc(dense_block_corr, dim * sizeof(IndiceT)));
  cudaMemset(*dense_block_corr, 0, dim * sizeof(IndiceT));
  CudaLaunchConfig config = GetCudaLaunchConfig(block_count, d);
  compute_dense_block_id<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, block_pointer_ids, *dense_block_corr);
}

template <typename dtype, typename itype>
__global__ void approxSparseDirectConv(CudaLaunchConfig config,
   const itype* __restrict__ in_block_ids, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_block_pointer, 
   const itype* __restrict__ in_block_pointer_ids, const itype* __restrict__ in_shape_ptr, 
   const itype* __restrict__ filter_ind_1d, const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape, const int* __restrict__ filter_segments_start, 
   const int* __restrict__ filter_segments_end, const int filter_segment_count,
   dtype* __restrict__ cout_conv_data, const itype* __restrict__ out_sh,
   int data_entry_count, int filter_weight_count, int data_dimension, int hypercube_size)
{
   __shared__ dtype accumulated_result[6125]; //TODO: dynamic allocation
  //1. get pointer offset:
  itype block_id = in_block_pointer_ids[blockIdx.x];
  itype* block_id_decompressed = new itype[data_dimension];
  decompress_block_id(block_id, in_shape_ptr, block_id_decompressed, data_dimension, hypercube_size);
  itype accumulated_offset = 0;
  index_KDto1D_(block_id_decompressed, in_shape_ptr, &accumulated_offset, data_dimension);

  //2. define neigbourhood (input halo) //TODO: parallelize && precompute
  itype neighbourhood_count = pow(data_dimension - 2, 3);
  itype* neighboring_blocks = new itype[neighbourhood_count];
  itype* mean_blocks = new itype[neighbourhood_count];
  for(size_t i = 0; i < neighbourhood_count; ++i){
    neighboring_blocks[i] = i;
  }
  itype* neighbourhood_shape = new itype[data_dimension];
  for(int i = 0; i < data_dimension - 2; ++i){
    neighbourhood_shape[i] = 3;
  }
  for(int i = data_dimension - 2; i < data_dimension; ++i){
    neighbourhood_shape[i] = 1;
  }
  for(int i = 0; i < neighbourhood_count; ++i){
    itype* neighbour_decompressed = new itype[data_dimension];
    index_1DtoKD(0, 0, neighboring_blocks[i], neighbourhood_shape, neighbour_decompressed, data_dimension);
    for(int j = 0; j < data_dimension; ++j){
      neighbour_decompressed[j] = neighbour_decompressed[j] - (neighbourhood_shape[i] - 1) / 2; //zero centering
    }
    //compress (like in_block_pointer_ids)
    itype val = 0;
    itype mul = 1;
    itype idx = 0;
    for(int j = data_dimension - 1; j >=0; --j) { //reorder dimensions to [dim1, ..., dimx, batch, channel] and compress
      if(j == data_dimension - 1){
        idx = data_dimension - 1;
        mul = mul * in_shape_ptr[data_dimension - 1];
      } else if(j == data_dimension - 2){
        val = val + mul * neighbour_decompressed[j];
        mul = mul * in_shape_ptr[0];
      } else {
        idx = j + 1;
        val = val + mul * floor(float(neighbour_decompressed[j]) / hypercube_size);
        mul = mul * in_shape_ptr[idx];
      }
    }
    neighboring_blocks[i] = block_id + val;
    delete[] neighbour_decompressed;
  }
  
  //3. find start and end of channels in blocks (neighbors and this)
  
  //4. find start and end of channels in filter

  //5. perform convolution in shared memory

  //6. if output allocated: write result to output

  delete[] block_id_decompressed;
  delete[] neighboring_blocks;
  delete[] mean_blocks;
  delete[] neighbourhood_shape;
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

  std::stringstream dout_s;
  //indices must! be sorted
  clock_t t;
  
  //preprocessing step (1) has to be performed only for one layer in the neural network! Also step 2 can be precomputed and shouldn't affect runtime of nn
  
  /////
  //1. Convert Coordinate List Format to sparse block format and compress k dimensional indices to 1d
  t = clock();
  int block_count = 0;
  const int hypercube_size = floor(pow(float(smpb / sizeof(T) / 2), 1. / filter_dim)); //compute block size: assumptions: i) all dimensions have the same size (not necessarly true); ii) two blocks per sm
  if(hypercube_size <= 0) return; //TODO: THROW ERROR
  IndiceT *in_block_ids = 0, *in_block_pointer = 0, *in_block_pointer_ids = 0;
  T *in_block_vals = 0;
  coo_to_blocks(context, d, i_ind.data(), i_val.data(), i_sh.data(), &in_block_ids, &in_block_vals, &in_block_pointer, &in_block_pointer_ids, data_dimension, data_entry_count, hypercube_size, block_count);
  LOG(INFO) << "Edge length: " << hypercube_size << " Shared memory per block: " << smpb << " sizeof T " << sizeof(T) << std::endl;
  dout_s << "t1: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;

  /////
  //2. prepare filter: directly manipulate 1D keys instead of kD indices and flip filter weights to be applicable for direct convolution and sort filter w.r.t. output and input channels
  t = clock();
  int *filter_segments_start = 0, *filter_segments_end = 0;
  int filter_segment_count_ = 0;
  T* filter_sorted_weights = 0;
  IndiceT* filter_sorted_ind_1d = 0;
  preprocess_filter(context, d, f_ind.data(), f_val.data(), f_sh.data(), i_sh.data(), data_dimension, filter_weight_count, &filter_segments_start, &filter_segments_end, &filter_sorted_weights, &filter_sorted_ind_1d, filter_segment_count_);
  dout_s << "t2: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;

  /////
  //3. generate dense lookup table for sparse block pointer ids to define a neighborhood
  IndiceT* dense_block_ids = 0;
  compute_dense_blocks<DeviceT, T, IndiceT>(d, i_sh.data(), in_block_pointer, in_block_pointer_ids, &dense_block_ids, data_dimension, data_entry_count, hypercube_size, block_count);

  /////
  //4. compute out shape
  //TODO
  IndiceT *out_sh = 0;
  checkCuda(cudaMalloc(&out_sh, data_dimension * sizeof(IndiceT)));
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //5. perform first convolution to know sparse output shape
  t = clock();
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  T* conv_res = 0;
  IndiceT conv_out_count = out_channel_count * 1;
  checkCuda(cudaMalloc(&conv_res, conv_out_count * sizeof(T)));
  cudaMemset(conv_res, 0, conv_out_count * sizeof(T));
  CudaLaunchConfig config_conv = GetCudaLaunchConfig(data_entry_count, d);
  approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
    in_block_ids, in_block_vals, in_block_pointer, in_block_pointer_ids, i_sh.data(),
    filter_sorted_ind_1d, filter_sorted_weights, f_sh.data(), filter_segments_start, filter_segments_end, filter_segment_count_, 
    conv_res, out_sh,  data_entry_count, filter_weight_count, data_dimension, hypercube_size);
  cudaDeviceSynchronize();
  dout_s << "t4: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;
  auto result_count = conv_out_count;

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
  cudaFree(out_sh);
  cudaFree(conv_res);
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
#define INIT_GPU_ALL(type) \
  INIT_GPU_TYPE(type, int64); \
  INIT_GPU_TYPE(type, int32);

INIT_GPU_ALL(float);
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
