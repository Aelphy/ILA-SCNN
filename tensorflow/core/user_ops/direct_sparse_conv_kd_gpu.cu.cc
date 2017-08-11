#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_conv_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

namespace tensorflow {

//Compress [batch, x, y, ...] indices into a [1D] key while keeping the data sorted. Except for [channel], which is handled seperately.
template <typename dtype>
__global__ void index_KDto1D(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* __restrict__ out_ind_ptr, dtype* __restrict__ out_channels_ptr, const int dimension_count, const int entry_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    dtype val = 0;
    dtype mul = 1;
    dtype idx = x;
    for(int i = dimension_count - 2; i >=0; --i) { //exclude channel
      idx = x * dimension_count +  i;
      val = val + mul * in_ptr[idx];
      mul = mul * in_shape_ptr[i];
    }
    const dtype channel = in_ptr[(x + 1)  * dimension_count - 1];
    out_ind_ptr[x] = val;
    out_channels_ptr[x] = channel;
  }
}

//decompress 1D key + channel into K dimensional indices
template <typename dtype>
__device__ void index_1DtoKD(const int x_in, const int x_out, const dtype in_index_1d, const dtype* __restrict__ in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    const dtype in_channel, dtype* __restrict__ out_ind_ptr, const int dimension_count){
  dtype idx_out = x_out * dimension_count;
  //1. compressed 1d key, except channel
  dtype *fact = new dtype[dimension_count - 1];
  fact[dimension_count - 2] = 1;
  for(int i = dimension_count - 3; i >= 0; i = i - 1){
    fact[i] = fact[i + 1] * in_shape_ptr[i + 1];
  }
  dtype r = in_index_1d;
  for(int i = 0; i < dimension_count - 1; ++i){
    out_ind_ptr[idx_out + i] = r / fact[i];
    r = r % fact[i];
  }
  delete[] fact;
  //2. add channel
  out_ind_ptr[idx_out + dimension_count - 1] = in_channel;
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
__global__ void unique_mask(CudaLaunchConfig config, const dtype* __restrict__ in_ptr, dtype* __restrict__ out_ptr){
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
              const itype* __restrict__ unique_count, dtype* __restrict__ unique_ptr, dtype* __restrict__ unique_cor){
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
__global__ void get_segment_start(CudaLaunchConfig config, itype* data_offset, const dtype* masked_indices, const dtype* unique_count){
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
__global__ void get_segment_end(CudaLaunchConfig config, itype* offset, const itype* __restrict__ segment_start, const dtype* __restrict__ count, const int filter_weight_count){
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
      //itype idy = data_dimension * f_id[y];
      itype idy = data_dimension * y;
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

//prepare filter weights
template <typename dtype>
__global__ void prepare_filter_weights_(CudaLaunchConfig config, 
                  const dtype* __restrict__ f_id_ptr, const dtype* __restrict__ f_sh_ptr, const dtype* __restrict__ in_sh_ptr,  
                  dtype* out_id_ptr, dtype* out_ch_ptr, dtype* in_channel, dtype* index, const int dimension_count, const int data_entry_count, const int filter_entry_count){
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
  const auto smpb = d.sharedMemPerBlock();

  std::stringstream dout_s;
  //indices must! be sorted

  clock_t t;
  /////
  //1. set channel to 0 and convert indices to 1D key
  // + define rule to work with more than one channel
  t = clock();
  IndiceT *in_ind_1d = 0;
  checkCuda(cudaMalloc(&in_ind_1d, data_entry_count * sizeof(IndiceT)));
  IndiceT *in_ind_1d_channels = 0;
  checkCuda(cudaMalloc(&in_ind_1d_channels, data_entry_count * sizeof(IndiceT)));
  CudaLaunchConfig config_i1d = GetCudaLaunchConfig(data_entry_count, d);
  index_KDto1D<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d,
      i_ind.data(), i_sh.data(),  in_ind_1d, in_ind_1d_channels, data_dimension, data_entry_count);
  cudaDeviceSynchronize();
  dout_s << "t1: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  /////
  //2. remove duplicates from data and apply stride/padding to obtain search structure
  t = clock();
  IndiceT *unique_masked = 0;
  checkCuda(cudaMalloc(&unique_masked, std::max(filter_weight_count, data_entry_count) * sizeof(IndiceT)));
  unique_mask<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d, in_ind_1d, unique_masked);
  IndiceT *unique_count = 0;
  checkCuda(cudaMalloc(&unique_count, std::max(filter_weight_count, data_entry_count) * sizeof(IndiceT)));
  CudaLaunchConfig config_1 = GetCudaLaunchConfig(1, d);
  compute_scan(context, d, unique_count, unique_masked, data_entry_count);
  IndiceT reduced_count = -1;
  cudaMemcpy(&reduced_count, unique_count + data_entry_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  IndiceT* reduced_indices = 0;
  checkCuda(cudaMalloc(&reduced_indices, reduced_count * sizeof(IndiceT)));
  IndiceT* reduced_correspondences = 0;
  checkCuda(cudaMalloc(&reduced_correspondences, reduced_count * sizeof(IndiceT)));
  unique_array<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d, in_ind_1d, unique_masked, unique_count, 
    reduced_indices, reduced_correspondences);
  cudaDeviceSynchronize();
  //  reduced_indices, in_ind_1d, data_dimension, data_entry_count, lower_search_limit, upper_search_limit);
  cudaDeviceSynchronize();
  dout_s << "t2: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  //TODO: apply stride/padding
  //TODO: initialize search structure
  
  /////
  //3. prepare filter: directly manipulate 1D keys instead of kD indices and flip filter weights to be applicable for direct convolution
  t = clock();
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
    f_ind.data(), f_sh.data(), i_sh.data(), filter_ind_1d, filter_out_channel, filter_in_channel, filter_id, data_dimension, reduced_count, filter_weight_count);
  cudaDeviceSynchronize();

  dout_s << "filter cout unsorted: ";
  std::vector<IndiceT> dout_sco(filter_weight_count);
  cudaMemcpy(&dout_sco[0], filter_out_channel, dout_sco.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_sco.size(); ++i) dout_s << dout_sco[i] << " ";
  dout_s << std::endl;

  dout_s << "filter cin unsorted: ";
  std::vector<IndiceT> dout_sci(filter_weight_count);
  cudaMemcpy(&dout_sci[0], filter_in_channel, dout_sci.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_sci.size(); ++i) dout_s << dout_sci[i] << " ";
  dout_s << std::endl;
 
 
  //sort filter w.r.t. output and input channels
  IndiceT* new_filter_indice = 0;
  IndiceT* filter_sorted_out = 0;
  IndiceT* filter_sorted_in = 0;
  IndiceT* filter_sorted_tmp_c_in = 0;
  IndiceT filter_segment_count = 0;
  int* filter_segments_start = 0;
  int* filter_segments_end = 0;
  checkCuda(cudaMalloc(&new_filter_indice, filter_weight_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&filter_sorted_out, filter_weight_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&filter_sorted_in, filter_weight_count * sizeof(IndiceT)));
  checkCuda(cudaMalloc(&filter_sorted_tmp_c_in, filter_weight_count * sizeof(IndiceT)));
  compute_sort(context, d, filter_out_channel, filter_sorted_out, filter_id, new_filter_indice, filter_weight_count);
  unique_mask<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_out, unique_masked); 
  compute_scan(context, d, unique_count, unique_masked, filter_weight_count);
  cudaMemcpy(&filter_segment_count, unique_count + filter_weight_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  checkCuda(cudaMalloc(&filter_segments_start, filter_weight_count * sizeof(int)));
  checkCuda(cudaMalloc(&filter_segments_end, filter_weight_count * sizeof(int)));
  cudaDeviceSynchronize();
  get_segment_start<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_segments_start, unique_masked, unique_count);
  cudaDeviceSynchronize();
  get_segment_end<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_segments_end, filter_segments_start, unique_count, filter_weight_count);
  cudaDeviceSynchronize();
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_tmp_c_in, filter_in_channel, new_filter_indice);
  cudaDeviceSynchronize();
  compute_segmented_sort(context, d, filter_sorted_tmp_c_in, filter_sorted_in, new_filter_indice, filter_id, filter_weight_count, filter_segment_count, filter_segments_start, filter_segments_end);
  cudaDeviceSynchronize();
  //apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_out_channel, filter_in_channel, filter_id);
  T* filter_sorted_weights = 0;
  checkCuda(cudaMalloc(&filter_sorted_weights, filter_weight_count * sizeof(T)));
  IndiceT* filter_sorted_ind_1d = 0;
  checkCuda(cudaMalloc(&filter_sorted_ind_1d, filter_weight_count * sizeof(IndiceT)));
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_weights, f_val.data(), filter_id);
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_ind_1d, filter_ind_1d, filter_id);
  dout_s << "t3: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  dout_s << "filter id sorted: ";
  std::vector<IndiceT> dout_sc(filter_weight_count);
  cudaMemcpy(&dout_sc[0], filter_sorted_ind_1d, dout_sc.size() * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_sc.size(); ++i) dout_s << dout_sc[i] << " ";
  dout_s << std::endl;
 
  dout_s << "filter id unsorted sorted: ";
  std::vector<IndiceT> dout_i(filter_weight_count);
  cudaMemcpy(&dout_i[0], filter_ind_1d, dout_i.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_i.size(); ++i) dout_s << dout_i[i] << " ";
  dout_s << std::endl;
  
  dout_s << "filter values sorted: ";
  std::vector<T> dout_inv(filter_weight_count);
  cudaMemcpy(&dout_inv[0], filter_sorted_weights, dout_inv.size() *sizeof(T), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_inv.size(); ++i) dout_s << dout_inv[i] << " ";
  dout_s << std::endl;
  
  dout_s << "filter values unsorted: ";
  std::vector<T> dout_id(filter_weight_count);
  cudaMemcpy(&dout_id[0], f_val.data(), dout_id.size() *sizeof(T), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_id.size(); ++i) dout_s << dout_id[i] << " ";
  dout_s << std::endl;
  
  dout_s << "filter sorted cout: ";
  std::vector<IndiceT> dout_invo(filter_weight_count);
  cudaMemcpy(&dout_invo[0], filter_sorted_out, dout_invo.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_invo.size(); ++i) dout_s << dout_invo[i] << " ";
  dout_s << std::endl;
  
  dout_s << "filter sorted cin: ";
  std::vector<IndiceT> dout_idi(filter_weight_count);
  cudaMemcpy(&dout_idi[0], filter_sorted_in, dout_idi.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_idi.size(); ++i) dout_s << dout_idi[i] << " ";
  dout_s << std::endl;

  dout_s << "tmp sorted ids: ";
  std::vector<IndiceT> dout_iditd(filter_weight_count);
  cudaMemcpy(&dout_iditd[0], new_filter_indice, dout_iditd.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_iditd.size(); ++i) dout_s << dout_iditd[i] << " ";
  dout_s << std::endl;

  dout_s << "tmp sorted cin: ";
  std::vector<IndiceT> dout_idit(filter_weight_count);
  cudaMemcpy(&dout_idit[0], filter_sorted_tmp_c_in, dout_idit.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_idit.size(); ++i) dout_s << dout_idit[i] << " ";
  dout_s << std::endl;

  dout_s << "filter id: ";
  std::vector<IndiceT> dout_idid(filter_weight_count);
  cudaMemcpy(&dout_idid[0], filter_id, dout_idid.size() *sizeof(IndiceT), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_idid.size(); ++i) dout_s << dout_idid[i] << " ";
  dout_s << std::endl;

  dout_s << "segments start: " << filter_segment_count << std::endl;
  std::vector<int> dout_si(filter_segment_count);
  cudaMemcpy(&dout_si[0], filter_segments_start, dout_si.size() *sizeof(int), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_si.size(); ++i) dout_s << dout_si[i] << " ";
  dout_s << std::endl;
  
  dout_s << "segments end: ";
  std::vector<int> dout_so(filter_segment_count);
  cudaMemcpy(&dout_so[0], filter_segments_end, dout_so.size() *sizeof(int), cudaMemcpyDeviceToHost);
  for(size_t i = 0; i < dout_so.size(); ++i) dout_s << dout_so[i] << " ";
  dout_s << std::endl;
  
  /////
  //4. compute out shape
  //TODO
  IndiceT *out_sh = 0;
  checkCuda(cudaMalloc(&out_sh, data_dimension * sizeof(IndiceT)));
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //5. perform approximated convolution
  t = clock();
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  T* conv_res = 0;
  IndiceT conv_out_count = out_channel_count * reduced_count;
  checkCuda(cudaMalloc(&conv_res, conv_out_count * sizeof(T)));
  cudaMemset(conv_res, 0, conv_out_count * sizeof(T));
  CudaLaunchConfig config_conv = GetCudaLaunchConfig(data_entry_count, d);
  approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
    i_ind.data(), i_val.data(), i_sh.data(), in_ind_1d, in_ind_1d_channels,
    f_ind.data(), f_val.data(), f_sh.data(), filter_ind_1d, filter_id,
    //f_ind.data(), filter_sorted_weights, f_sh.data(), filter_sorted_ind_1d, filter_id,
    reduced_indices, reduced_count, //binary search
    conv_res, out_sh,
    data_entry_count, filter_weight_count, data_dimension);
  cudaDeviceSynchronize();
  dout_s << "t5: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  /////
  //6. remove zero entries and convert from keys to indices
  t = clock();
  IndiceT *non_zero_masked = 0;
  checkCuda(cudaMalloc(&non_zero_masked, conv_out_count * sizeof(IndiceT)));
  CudaLaunchConfig config_r1d = GetCudaLaunchConfig(conv_out_count, d);
  non_zero_mask<<<config_r1d.block_count, config_r1d.thread_per_block, 0, d.stream()>>>(config_r1d, conv_res, non_zero_masked);
  IndiceT *non_zero_count = 0;
  checkCuda(cudaMalloc(&non_zero_count, conv_out_count * sizeof(IndiceT)));
  compute_scan(context, d, non_zero_count, non_zero_masked, conv_out_count);
  IndiceT result_count = 0;
  cudaMemcpy(&result_count, non_zero_count + conv_out_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  dout_s << "t6: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;
  
  /////
  //7. Create and fill output tensor
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
  result_to_output<<<config_r1d.block_count, config_r1d.thread_per_block, 0, d.stream()>>>(config_r1d, 
    reduced_correspondences, i_ind.data(), non_zero_count, conv_res, reduced_count, data_dimension, o_ind.data(), o_val.data());
  cudaMemcpy(o_sh.data(), out_sh, data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  dout_s << "t7: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  //# free memory
  t = clock();
  cudaFree(in_ind_1d_channels);
  cudaFree(in_ind_1d);
  cudaFree(unique_masked);
  cudaFree(unique_count);
  cudaFree(reduced_indices);
  cudaFree(reduced_correspondences);
  cudaFree(filter_segments_start);
  cudaFree(filter_segments_end);
  cudaFree(filter_ind_1d);
  cudaFree(filter_id);
  cudaFree(filter_out_channel);
  cudaFree(filter_in_channel);
  cudaFree(filter_sorted_tmp_c_in);
  cudaFree(new_filter_indice);
  cudaFree(filter_sorted_out);
  cudaFree(filter_sorted_in);
  cudaFree(out_sh);
  cudaFree(conv_res);
  cudaFree(non_zero_masked);
  cudaFree(non_zero_count);
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
