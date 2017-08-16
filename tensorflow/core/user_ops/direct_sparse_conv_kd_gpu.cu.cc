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

//decompress 1D key + channel into K dimensional indices
template <typename dtype>
__device__ void index_1DtoKD(const int x_in, const int x_out, const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    const dtype in_channel, dtype* __restrict__ out_ind_ptr, const int dimension_count){
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
                    dtype* out_ind_ptr, dtype* out_id_ptr, const int dimension_count, const int entry_count, const int voxel_size){
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
        auto inval = in_ptr[idx];
        val = val + mul * inval;
        mul = mul * in_shape_ptr[dimension_count - 1];
      } else if(i == dimension_count - 2){
        idx = x * dimension_count;
        val = val + mul * in_ptr[idx];
        mul = mul * in_shape_ptr[0];
      } else {
        ii = i + 1;
        idx = x * dimension_count + ii;
        val = val + mul * floor(float(in_ptr[idx]) / voxel_size);
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
              const itype* __restrict__ unique_count, dtype*  unique_cor){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(unique_masked_ptr[x] == 1){
      unique_cor[unique_count[x] - 1] = x;
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

template <typename dtype, typename itype>
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

template<typename DeviceT, typename T, typename IndiceT> inline void
input_to_blocks(  OpKernelContext* context, DeviceT d, const IndiceT* in_ids_kd, const T* in_vals, const IndiceT* in_shape, IndiceT** block_ids_1d, T** block_vals,
                  IndiceT** pointer_ids, int dimension_count, int data_entry_count, int voxel_size, int& block_count)
{
  std::stringstream dout_s;
  IndiceT *tmp_data = 0, *tmp_data2 = 0;
  IndiceT *sorted_voxel_ids = 0, *sorted_id = 0;
  checkCuda(cudaMalloc(&tmp_data, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&tmp_data2, data_entry_count * sizeof(IndiceT)));
  CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
  auto &voxel_id = tmp_data;
  auto &data_id = tmp_data2;
  compute_voxel_id1D<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, in_shape, voxel_id, data_id, dimension_count, data_entry_count, voxel_size); 
  cudaDeviceSynchronize();
  //1. put entries per block into consecutive segments of a list
  checkCuda(cudaMalloc(&sorted_voxel_ids, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&sorted_id, data_entry_count * sizeof(IndiceT)));
  compute_sort(context, d, voxel_id, sorted_voxel_ids /*sorted voxel ids*/, data_id, sorted_id /*sorted data ids*/, data_entry_count);
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
  checkCuda(cudaMalloc(pointer_ids, block_count_ * sizeof(IndiceT)));
  compute_block_start<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, unique_mask, unique_count, *pointer_ids);
  cudaDeviceSynchronize();
  //apply block structure to data
  auto &id1d = tmp_data;
  index_KDto1D<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, in_shape, id1d, dimension_count, data_entry_count);
  checkCuda(cudaMalloc(block_ids_1d, data_entry_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(block_vals, data_entry_count * sizeof(T)));
  apply_sorted_indices<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, *block_ids_1d, id1d, sorted_id);
  apply_sorted_indices<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, *block_vals, in_vals, sorted_id);
  
  dout_s << "i" << std::endl;
  std::vector<IndiceT> dout_ri(data_entry_count);
  cudaMemcpy(&dout_ri[0], sorted_id, dout_ri.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_ri.size(); ++i){
    dout_s << dout_ri[i] << " ";
  }
  dout_s << std::endl;
  LOG(INFO) << dout_s.str() << std::endl;

  //# free temporary resources 
  cudaDeviceSynchronize();
  cudaFree(tmp_data);
  cudaFree(tmp_data2);
  cudaFree(sorted_voxel_ids);
  cudaFree(sorted_id);
}

//preprocess_filter(context, d, f_ind.data(), f_val.data(), f_sh.data(), i_sh.data(), data_dimension, filter_weight_count);
//(tensorflow::CudaLaunchConfig, const float *, const tensorflow::int64 *, const tensorflow::int64 *, tensorflow::int64 *, tensorflow::int64 *, tensorflow::int64 *, tensorflow::int64 *, int, int)
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
  /////
  //1. Convert Coordinate List Format to sparse block format and compress k dimensional indices to 1d
  t = clock();
  int block_count = 0;
  const int voxel_size = 2;
  //const int voxel_size = floor(pow(float(smpb / sizeof(T)), 1. / filter_dim)); //compute block size: assumption: all dimensions have the same size (not necessarly true)
  IndiceT *in_block_ids = 0, *in_block_pointer_ids = 0;
  T *in_block_vals = 0;
  input_to_blocks(context, d, i_ind.data(), i_val.data(), i_sh.data(), &in_block_ids, &in_block_vals, &in_block_pointer_ids, data_dimension, data_entry_count, voxel_size, block_count);
  LOG(INFO) << "Edge length: " << voxel_size << " Shared memory per block: " << smpb << " sizeof T " << sizeof(T) << std::endl;
  dout_s << "t1: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  dout_s << "input indices:  value" << std::endl;
  std::vector<IndiceT> dout_ri(data_entry_count * data_dimension);
  std::vector<T> dout_rv(data_entry_count);
  cudaMemcpy(&dout_ri[0], i_ind.data(), dout_ri.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  cudaMemcpy(&dout_rv[0], i_val.data(), dout_rv.size() *sizeof(T), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < data_entry_count; ++i){
    for(size_t j = 0; j < data_dimension; ++j){
      auto idx = data_dimension * i + j;
      dout_s << dout_ri[idx] << " ";
    }
    dout_s << ": " << dout_rv[i] << std::endl;
  }
  dout_s << std::endl;

  dout_s << "block indices:  value" << std::endl;
  std::vector<IndiceT> dout_bi(data_entry_count);
  std::vector<T> dout_bv(data_entry_count);
  cudaMemcpy(&dout_bi[0], in_block_ids, dout_bi.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  cudaMemcpy(&dout_bv[0], in_block_vals, dout_bv.size() *sizeof(T), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < data_entry_count; ++i){
    dout_s << dout_bi[i] << ": " << dout_bv[i] << std::endl;
  }
  dout_s << std::endl;
  
  dout_s << "block segments " << block_count << std::endl;
  std::vector<IndiceT> dout_bb(block_count);
  cudaMemcpy(&dout_bb[0], in_block_pointer_ids, dout_bb.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_bb.size(); ++i){
    dout_s << dout_bb[i] << std::endl;
  }
  dout_s << std::endl;

  /////
  //2. prepare filter: directly manipulate 1D keys instead of kD indices and flip filter weights to be applicable for direct convolution and sort filter w.r.t. output and input channels
  t = clock();
  int* filter_segments_start = 0;
  int* filter_segments_end = 0;
  int filter_segment_count_;
  T* filter_sorted_weights = 0;
  IndiceT* filter_sorted_ind_1d = 0;
  preprocess_filter(context, d, f_ind.data(), f_val.data(), f_sh.data(), i_sh.data(), data_dimension, filter_weight_count, &filter_segments_start, &filter_segments_end, &filter_sorted_weights, &filter_sorted_ind_1d, filter_segment_count_);
  dout_s << "t2: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  /////
  //3. compute out shape
  //TODO
  IndiceT *out_sh = 0;
  checkCuda(cudaMalloc(&out_sh, data_dimension * sizeof(IndiceT)));
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //4. perform first convolution to know sparse output shape
  t = clock();
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  T* conv_res = 0;
  IndiceT conv_out_count = out_channel_count * 1;
  checkCuda(cudaMalloc(&conv_res, conv_out_count * sizeof(T)));
  cudaMemset(conv_res, 0, conv_out_count * sizeof(T));
  CudaLaunchConfig config_conv = GetCudaLaunchConfig(data_entry_count, d);
  /*approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
    i_ind.data(), i_val.data(), i_sh.data(), in_ind_1d, in_ind_1d_channels,
    //f_ind.data(), f_val.data(), f_sh.data(), filter_ind_1d, filter_id,
    f_ind.data(), filter_sorted_weights, f_sh.data(), filter_sorted_ind_1d, filter_id,
    reduced_indices, reduced_count, //binary search
    conv_res, out_sh,
    data_entry_count, filter_weight_count, data_dimension);*/
  cudaDeviceSynchronize();
  dout_s << "t4: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  auto result_count = conv_out_count;
  /////
  //5. Create output tensor and fill it in a second run of convolution
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
#define INIT_GPU_TYPE(type, indice_type)      \
 template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, type, indice_type>;
#define INIT_GPU_ALL(type) \
  INIT_GPU_TYPE(type, int64); \
  INIT_GPU_TYPE(type, int32);

INIT_GPU_ALL(float);
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
