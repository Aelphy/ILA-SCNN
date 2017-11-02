#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_conv_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {

template <typename dtype, int data_dimension> __device__ __forceinline__ void 
map_index_kd_to_shared_id(const dtype* __restrict__ in_id, const dtype* __restrict__ filter_id, const dtype* __restrict__ block_start_array,
   const dtype* __restrict__ f_shape, dtype* out_id, const int hypercube_size)
{
  int id1d = 0;
  int mul = 1;
  for(int i = 0; i < data_dimension - 2; ++i){
    int id1d_val = (in_id[i] - filter_id[i] + (f_shape[i] - 1) - block_start_array[i]);
    id1d += id1d_val * mul;
    mul = mul * (hypercube_size + f_shape[i] - 1); //TODO: precompute?
  }
  *out_id = id1d;
}

template <typename dtype, int data_dimension> __device__ __forceinline__ void 
map_shared_to_channel_buffer(const dtype in_id, const dtype* block_start_array, const dtype* f_shape,
   const dtype* __restrict__ out_shape_ptr, dtype* out_index, const int hypercube_size)
{
  dtype fact[data_dimension- 2];
  fact[0] = 1;
  for(int i = 1; i < data_dimension - 2; ++i){
    fact[i] = fact[i - 1] * (hypercube_size + f_shape[i - 1] - 1);
  }
  dtype r =  in_id;
  dtype out_mul = 1;
  dtype out_val = 0;
  for(int i = data_dimension - 3; i >= 0; --i){ //TODO: check index (++i, --i)?
    dtype id = r / fact[i];
    r = r % fact[i];
    dtype val = id + block_start_array[i] - (f_shape[i] - 1) / 2;
    auto out_limit = out_shape_ptr[i + 1];
    if(val < 0 || val >= out_limit){
      out_val = -1; //index out of tensor bounds
      break;
    }
    out_val = out_val + val * out_mul;
    out_mul = out_mul * out_shape_ptr[i + 1];
  }
  *out_index = out_val;
}

//index war :)
template <typename dtype, int data_dimension> __device__ __forceinline__ void 
map_channel_buffer_1d_to_kd(dtype in_id, const dtype* out_shape_ptr, dtype* out_index)
{
  dtype fact[data_dimension- 2];
  fact[0] = 1;
  for(int i = 2; i < data_dimension; ++i){
    fact[i - 1] = fact[i - 2] * out_shape_ptr[i - 1];
  }
  dtype r = in_id;
  for(int i =  data_dimension - 3; i >= 0; --i){
    out_index[i] = r / fact[i];
    r = r % fact[i];
  }
}

template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK) 
decompress_1d_to_kd(CudaLaunchConfig config, const dtype* in_data_id1d, const dtype* in_filter_id1d, const dtype* in_shape_ptr, const dtype* filter_shape_ptr, dtype* out_data_idkd, dtype* out_filter_idkd, int data_start, int data_end, int filter_start, int filter_end)
{
  int data_size = data_end - data_start;
  int filter_size = filter_end - filter_start;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    if(x < data_size){
      int id = x;
      index_1DtoKD_reduced<dtype, data_dimension>(0, in_data_id1d[id + data_start], in_shape_ptr, &out_data_idkd[(data_dimension - 2) * id], 1);
    } else {
      int id = x - data_size;
      index_1DtoKD_reduced<dtype, data_dimension>(0, in_filter_id1d[id + filter_start], filter_shape_ptr, &out_filter_idkd[(data_dimension - 2) * id], 0);
    }
  }
}

//(int *, int *, int *, const unsigned int, const int *, const int *, float [768], int, int, int)
template <typename dtype> __device__ __forceinline__ void 
conv_init_parameters(int* filter_start, int* filter_end, int* block_start_, const unsigned int block_id, const int* input_block_mapping, const int* filter_channel_mapping, dtype* accumulated_result, int batch, int in_channel_count, int out_channel)
{
  int offbix = batch * in_channel_count;
  int bid = input_block_mapping[offbix] + block_id;
  for(int x = threadIdx.x; x < in_channel_count; x += blockDim.x){
    if(bid >= input_block_mapping[x + offbix] && bid < input_block_mapping[x + 1 + offbix]){
      accumulated_result[0] = x;
    }
  }
  __syncthreads();
  int channel = accumulated_result[0];
  *block_start_ = bid;
  int offfc = out_channel * in_channel_count + channel;
  *filter_start = filter_channel_mapping[offfc];
  *filter_end = filter_channel_mapping[offfc + 1];
}

template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
count_non_zeros_buffer(CudaLaunchConfig config, const dtype* __restrict__ in_buffer, int* out_var /*initialized to zero*/, int batch_id, int channel_id, int out_channel_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(in_buffer[x] != 0){
      atomicAdd(&(out_var[channel_id + batch_id * out_channel_count]), 1);
    }
  }
}

//TODO: thread divergence... (find better approach with less idle threads)!
template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
count_non_zeros_dense(CudaLaunchConfig config, const dtype* __restrict__ in_buffer, int* dense_out_count /*initialized to zero*/, 
   int* result_block_count, const itype* out_shape_ptr, int hypercube_size){
  itype idx[data_dimension - 2];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    if(in_buffer[x] != 0){
      map_channel_buffer_1d_to_kd<itype, data_dimension>((itype) x, out_shape_ptr, idx);
      for(int i = 0; i < data_dimension - 2; ++i){
        idx[i] = floor(float(idx[i]) / hypercube_size);
      }
      int mul = 1;
      int idx_out = 0;
      for(int i = 0; i < data_dimension - 2; ++i){ //TODO: check
        idx_out += idx[i] * mul;
        mul = mul * ceil(out_shape_ptr[i] / float(hypercube_size));
      }
      int old_val = atomicAdd(&dense_out_count[idx_out], 1);
      if(old_val == 0){
        atomicAdd(result_block_count, 1);
      }
    }
  }
}

//TODO: thread divergence... (find better approach with less idle threads)!
template <typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
write_conv_res(CudaLaunchConfig config, const dtype* __restrict__ in_buffer, int* dense_out_count, int* dense_out_offset, int* channel_offset,
   itype* out_id, dtype* out_val, const itype* out_shape_ptr, int channel, int entries_per_channel, int hypercube_size, int batch, int out_channel_count)
{
  itype idx_all[data_dimension];
  itype block_idx[data_dimension - 2];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if(in_buffer[x] != 0){
      idx_all[0] = batch;
      idx_all[data_dimension - 1] = channel;
      itype* idx = &idx_all[1];
      map_channel_buffer_1d_to_kd<itype, data_dimension>((itype) x, out_shape_ptr, idx);
      for(int i = 0; i < data_dimension - 2; ++i){
        block_idx[i] = floor(float(idx[i]) / hypercube_size);
      }
      int mul = 1;
      int block_idx_out = 0;
      for(int i = 0; i < data_dimension - 2; ++i){ //TODO: check
        block_idx_out += block_idx[i] * mul;
        mul = mul * ceil(out_shape_ptr[i] / float(hypercube_size));
      }
      int block_idx_1d = atomicAdd(&dense_out_count[block_idx_out], 1);
      int idx_out = block_idx_1d +  dense_out_offset[block_idx_out] + channel_offset[channel + batch * out_channel_count];

      out_val[idx_out] = in_buffer[x];
      itype out_id_1d;
      index_KDto1D_<itype, data_dimension>(idx_all, out_shape_ptr, &out_id_1d);
      out_id[idx_out] =  out_id_1d;
    }
  }
}

//TODO: thread divergence... (find better approach with less idle threads)!
template <typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
write_conv_res2(CudaLaunchConfig config, const itype* in_buffer, const itype* in_ids, itype* out_ind, dtype* out_val, const itype* out_shape_ptr, int out_entries, int channel, int batch, int* count, int* offset_)
{
  itype offset = *offset_;
  itype idkd[data_dimension];
  idkd[0] = batch;
  idkd[data_dimension - 1] = channel;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    itype idx = in_ids[x];
    dtype outval = dtype(in_buffer[x]) / 2048;
    itype* idkd_r = &idkd[1];
    //map_channel_buffer_1d_to_kd<itype, data_dimension>(idx, out_shape_ptr, idkd_r);
    itype fact[data_dimension- 2]; 
    fact[0] = 1;
    for(int i = 2; i < data_dimension; ++i){
      fact[i - 1] = fact[i - 2] * out_shape_ptr[i - 1]; 
    }
    itype r = idx;
    for(int i =  data_dimension - 3; i >= 0; --i){
      idkd_r[i] = r / fact[i];
      r = r % fact[i];
    }
   
 
    itype out_id_1d = 0;
    index_KDto1D_<itype, data_dimension>(&idkd[0], out_shape_ptr, &out_id_1d);
    int out_x = atomicAdd(count, 1);
    out_val[out_x] = outval;
    out_ind[out_x] = out_id_1d;
  }
}

template <typename dtype, typename itype, typename btype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
abs_values(CudaLaunchConfig config, const dtype* __restrict__ in_buffer, btype* out_buffer, itype* out_ids, int* out_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(in_buffer[x] == 0) continue;
    int out_x = atomicAdd(out_count, 1);
    out_buffer[out_x] = itype(abs(in_buffer[x] * 2048));
    out_ids[out_x] = x;
  }
}

template <typename dtype, typename itype, typename btype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
relu_values(CudaLaunchConfig config, const dtype* __restrict__ in_buffer, btype* out_buffer, itype* out_ids, int* out_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(in_buffer[x] <= 0) continue;
    int out_x = atomicAdd(out_count, 1);
    out_buffer[out_x] = itype(in_buffer[x] * 2048);
    out_ids[out_x] = x;
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
gmSparseDirectConv(Cuda2DLaunchConfig config, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_shape_ptr, 
  const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape_ptr, const itype* __restrict__ out_sh, dtype* dense_channel_buffer,
  const itype* __restrict__ filter_ids_kd, const itype* __restrict__ input_ids_kd, int batch, int block_id_start, int block_id_end, int filter_start, int filter_end)
{
  //1. define variables needed for convolution (overhead)
  const int block_start = block_id_start;
  const int block_end = block_id_end;
  const int filter_weight_start = filter_start;
  const int filter_weights_end = filter_end;
  const int ch_filter_weight_count = filter_weights_end - filter_weight_start;
  const int input_data_count = block_end - block_start;
  const int operation_count = ch_filter_weight_count * input_data_count;
  //load data to registers
  itype filter_shape[data_dimension];
  itype input_shape[data_dimension];
  for(int i = 0; i < data_dimension; ++i){
    filter_shape[i] = filter_shape_ptr[i];
  }
  for(int i = 0; i < data_dimension; ++i){
    input_shape[i] = in_shape_ptr[i];
  }
  itype out_id[data_dimension - 2];
  //2. perform convolution with kd indices
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      const int fid = y;
      const int did = x;
      const itype* data_index_kd = &input_ids_kd[(did + block_start) * (data_dimension - 2)];
      const itype* filter_index_kd = &filter_ids_kd[(fid + filter_weight_start) * (data_dimension - 2)];
      const int block_data_id1d = did + block_start;
      const int filter_data_id1d = fid + filter_weight_start;
      const dtype fv = filter_weights[filter_data_id1d];
      const dtype iv = in_block_vals[block_data_id1d];
      const dtype out_v = fv * iv;
      itype acc_id = 0;
      bool is_valid = true;
      int mul = 1;
      for(int i = 0; i < data_dimension - 2; ++i){
        out_id[i] = data_index_kd[i] - filter_index_kd[i] + (filter_shape[i] - 1) / 2;
        if(out_id[i] < 0 || out_id[i] >= input_shape[i + 1]){
          is_valid = false;
          break;
        }
        acc_id += out_id[i] * mul;
        mul = mul * input_shape[i + 1];
      }
      //if(global_acc_id >= 0){
      if(is_valid){
        atomicAdd(&(dense_channel_buffer[acc_id]), out_v);
      }
    }
  }
}


namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConvFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, float max_density, const std::string& filter_type) const {
  clock_t t;
  const Tensor *in_indices, *in_values, *in_shape, *in_block_ptr_t, *data_count_t, *filter_indices, *filter_values, *filter_shape, *filter_channel_mapping_t;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  OP_REQUIRES_OK(context, context->input("filter_channel_mapping", &filter_channel_mapping_t));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto f_sh = filter_shape->flat<IndiceT>();
  auto f_ind = filter_indices->flat<IndiceT>();
  auto f_val = filter_values->flat<T>(); 
  auto i_mapping = in_block_ptr_t->flat<int>();
  auto f_mapping = filter_channel_mapping_t->flat<int>();
  int data_entry_count, filter_weight_count;
  cudaMemcpy(&data_entry_count, i_mapping.data() + i_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&filter_weight_count, f_mapping.data() + f_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  std::vector<IndiceT> cpu_in_shape(data_dimension);
  cudaMemcpy(&cpu_in_shape[0], i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  IndiceT channel_dense_size = 1;
  int max_dim = 0;
  for(size_t i = 0; i < data_dimension - 2; ++i){
    channel_dense_size = (channel_dense_size * cpu_in_shape[i + 1]); //x,y,z, ... (no channel and no batch) rounded up
  }
  const IndiceT batch_count = cpu_in_shape[0];
  const IndiceT in_channel_count = cpu_in_shape[data_dimension - 1];
  const IndiceT batch_dense_size = in_channel_count * batch_count;
  const IndiceT tensor_dense_size = channel_dense_size * batch_dense_size;

  //LOG(INFO) << "conv " << data_entry_count << " " << double(data_entry_count) / tensor_dense_size << " : " << cpu_in_shape[0] << " " << cpu_in_shape[1] << " " << cpu_in_shape[2] << " " << cpu_in_shape[3] << " " << cpu_in_shape[4] << std::endl;
  const IndiceT *in_block_ids = i_ind.data();
  const T *in_block_vals = i_val.data();
  const int* input_block_mapping = in_block_ptr_t->flat<int>().data();
  std::vector<int> cpu_input_block_mapping(batch_count * in_channel_count + 1);
  cudaMemcpy(&cpu_input_block_mapping[0], input_block_mapping, (batch_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  Tensor fcm_tensor, fss_tensor, fse_tensor, fsw_tensor, fsi_tensor;
  const int* filter_channel_mapping = filter_channel_mapping_t->flat<int>().data();
  const T* filter_sorted_weights = f_val.data();
  const IndiceT* filter_sorted_ind_1d = f_ind.data();
  std::vector<int> cpu_filter_channel_mapping(out_channel_count * in_channel_count + 1);
  cudaMemcpy(&cpu_filter_channel_mapping[0], filter_channel_mapping, (out_channel_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  /////
  //1. Compute out shape
  //TODO
  Tensor out_sh_tensor;
  IndiceT *out_sh = 0;
  allocate_tensor(context, out_sh_tensor, &out_sh,  data_dimension);
  cudaMemcpy(out_sh, i_sh.data(), (data_dimension - 1) * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaMemcpy(out_sh + data_dimension - 1, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //2. decompress 1d to kd indice in temporary buffer
  //TODO: is a better approach possible? (decompression in kernels)
  t = clock(); 
  Tensor in_id_kd_buffer, filter_id_kd_buffer;
  IndiceT *in_id_kd_ptr = 0, *filter_id_kd_ptr = 0;
  allocate_tensor(context, in_id_kd_buffer, &in_id_kd_ptr, (data_dimension - 2) * data_entry_count);
  allocate_tensor(context, filter_id_kd_buffer, &filter_id_kd_ptr, (data_dimension - 2) * filter_weight_count);
  CudaLaunchConfig config_dec = GetCudaLaunchConfig(data_entry_count + filter_weight_count, d);
  decompress_1d_to_kd<IndiceT, data_dimension><<<config_dec.block_count, config_dec.thread_per_block, 0, d.stream()>>>(config_dec, in_block_ids, filter_sorted_ind_1d, i_sh.data(), f_sh.data(), in_id_kd_ptr, filter_id_kd_ptr, 0, data_entry_count, 0, filter_weight_count);
  cudaStreamSynchronize(d.stream());

  /////
  //3. Perform convolution; upper bound of output shape is known by max_density
  t = clock();
  CudaLaunchConfig config_buffer = GetCudaLaunchConfig(channel_dense_size, d);
  CudaLaunchConfig config_rbuffer = GetCudaLaunchConfig(channel_dense_size * max_density, d);
  Tensor channel_offset_tensor, result_block_count_tensor, result_dense_count_tensor;
  
  const IndiceT result_count = ceil(tensor_dense_size * max_density);
  const int max_channel_count = floor(result_count / double(out_channel_count * batch_count));
  int *result_block_count, *result_dense_count;
  allocate_tensor(context, result_block_count_tensor, &result_block_count, 1);
  allocate_tensor(context, result_dense_count_tensor, &result_dense_count, 1);
  cudaMemset(result_block_count, 0, sizeof(int)); //stores the dense result of the computed output channel in buffer
  Tensor channel_buffer_tensor, abs_channel_buffer_tensor, in_channel_ids_tensor, sorted_channel_ids_tensor, tmp_channel_tensor;
  T* channel_buffer = 0;
  IndiceT  *in_channel_ids_buffer = 0, *sorted_channel_ids_buffer = 0, *abs_channel_buffer, *tmp_channel_buffer = 0;
  allocate_tensor(context, channel_buffer_tensor, &channel_buffer, channel_dense_size);
  allocate_tensor(context, abs_channel_buffer_tensor, &abs_channel_buffer, channel_dense_size);
  allocate_tensor(context, in_channel_ids_tensor, &in_channel_ids_buffer, channel_dense_size);
  allocate_tensor(context, sorted_channel_ids_tensor, &sorted_channel_ids_buffer, channel_dense_size);
  allocate_tensor(context, tmp_channel_tensor, &tmp_channel_buffer, channel_dense_size);
  Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL, *data_count = NULL;
  TensorShape out_ind_shape = {(IndiceT) result_count, (IndiceT) 1};
  TensorShape out_val_shape = {(IndiceT) result_count};
  TensorShape out_sh_shape = {(IndiceT) data_dimension};
  TensorShape out_count_shape = {(IndiceT) (batch_count * out_channel_count + 1)};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  OP_REQUIRES_OK(context, context->allocate_output("out_block_channel_mapping", out_count_shape, &data_count));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->matrix<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto data_offset = data_count->flat<int>();
  int* data_offset_ptr = data_offset.data();
  cudaMemset(data_offset_ptr, 0, (batch_count * out_channel_count + 1) * sizeof(int)); //stores the dense result of the computed output channel in buffer
  for(int i = 0; i < batch_count; ++i){
    for(int j = 0; j < out_channel_count; ++j){
      cudaStreamSynchronize(d.stream());
      cudaMemset(channel_buffer, 0, channel_dense_size * sizeof(T)); //stores the dense result of the computed output channel in buffer
      for(int k = 0; k < in_channel_count; ++k){
        int block_start_ = cpu_input_block_mapping[i * in_channel_count + k];
        int block_end_ = cpu_input_block_mapping[i * in_channel_count + k + 1]; //all blocks for batch
        int filter_start = cpu_filter_channel_mapping[j * in_channel_count + k];
        int filter_end = cpu_filter_channel_mapping[j * in_channel_count + k + 1];
        int data_block_count = block_end_ - block_start_;
        int op_count = (filter_end - filter_start) * data_block_count;
        if(op_count <= 0) continue;
        // int filter_weight_count, int hypercube_size_, int filter_max_dim, int batch, int block_id_start, int block_id_end, int filter_start, int filter_end
        Cuda2DLaunchConfig config_conv_ = GetCuda2DLaunchConfig(data_block_count, filter_end - filter_start, d);
        gmSparseDirectConv<T, IndiceT, data_dimension><<<config_conv_.block_count, config_conv_.thread_per_block, 0, d.stream()>>>(config_conv_,
                 in_block_vals, i_sh.data(),
                 filter_sorted_weights, f_sh.data(),
                 out_sh, channel_buffer, filter_id_kd_ptr, in_id_kd_ptr,
                 i, block_start_, block_end_, filter_start, filter_end);
      }
      cudaMemset(result_dense_count, 0, sizeof(int));
      cudaStreamSynchronize(d.stream());
      if(filter_type == "K-ABS"){
        abs_values<<<config_buffer.block_count, config_buffer.thread_per_block, 0, d.stream()>>>(config_buffer, channel_buffer, abs_channel_buffer, in_channel_ids_buffer, result_dense_count);
      } else if(filter_type == "K-RELU"){
        relu_values<<<config_buffer.block_count, config_buffer.thread_per_block, 0, d.stream()>>>(config_buffer, channel_buffer, abs_channel_buffer, in_channel_ids_buffer, result_dense_count);
      } else {
        //LOG(ERROR) << "unsupported filter type: " << filter_type << ". Only 'K-ABS' and 'K-RELU' are supported.";
        return;
      }
      cudaStreamSynchronize(d.stream());
      int cpu_result_count;
      IndiceT* write_ids;
      IndiceT* write_vals;
      int start_offset = 0; //offset to handle increasingly sorted values (interested in decreasingly sorted values)
      cudaMemcpy(&cpu_result_count, result_dense_count, sizeof(int), cudaMemcpyDeviceToHost);
      if(cpu_result_count >= max_channel_count){
        compute_sort(context, d, abs_channel_buffer, tmp_channel_buffer, in_channel_ids_buffer, sorted_channel_ids_buffer, cpu_result_count); //sorted increasing
        cudaStreamSynchronize(d.stream());
        write_ids = sorted_channel_ids_buffer;
        write_vals = tmp_channel_buffer;
        start_offset = cpu_result_count - max_channel_count;
      } else {
        write_ids = in_channel_ids_buffer;
        write_vals = abs_channel_buffer;
      }
      if(cpu_result_count > 0){
        CudaLaunchConfig config_dbuffer_ = GetCudaLaunchConfig(min(cpu_result_count, max_channel_count), d);
        if(config_dbuffer_.virtual_thread_count > 0){
          write_conv_res2<T, IndiceT, data_dimension><<<config_dbuffer_.block_count, config_dbuffer_.thread_per_block, 0, d.stream()>>>(config_dbuffer_, write_vals + start_offset, 
              write_ids + start_offset, o_ind.data(), o_val.data(), i_sh.data(), channel_dense_size * max_density, j, i, result_block_count, data_offset_ptr);
          cudaStreamSynchronize(d.stream());
        }
      }
      data_offset_ptr = data_offset.data() + i * out_channel_count + j + 1; //TODO
      cudaMemcpy(data_offset_ptr, result_block_count, sizeof(int), cudaMemcpyDeviceToDevice);
     }
  }
  cudaMemcpy(o_sh.data(), out_sh, data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
}


template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK) 
decompress_1d_to_kd(CudaLaunchConfig config, const dtype* in_data_id1d, const dtype* in_filter_id1d, const dtype* output_id1d, const dtype* in_shape_ptr, const dtype* filter_shape_ptr, const dtype* out_shape_ptr, dtype* out_data_idkd, dtype* out_filter_idkd, dtype* out_out_idkd, int data_start, int data_end, int filter_start, int filter_end, int output_start, int output_end)
{
  int data_size = data_end - data_start;
  int filter_size = filter_end - filter_start;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    if(x < data_size){
      int id = x;
      index_1DtoKD_reduced<dtype, data_dimension>(0, in_data_id1d[id + data_start], in_shape_ptr, &out_data_idkd[(data_dimension - 2) * id], 1);
    } else if(x < data_size + filter_size){
      int id = x - data_size;
      index_1DtoKD_reduced<dtype, data_dimension>(0, in_filter_id1d[id + filter_start], filter_shape_ptr, &out_filter_idkd[(data_dimension - 2) * id], 0);
    } else {
      int id = x - data_size - filter_size;
      index_1DtoKD_reduced<dtype, data_dimension>(0, output_id1d[id + output_start], out_shape_ptr, &out_out_idkd[(data_dimension - 2) * id], 1);
    }
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
gmSparseDirectConvBackProp(Cuda2DLaunchConfig config, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_shape_ptr, 
  const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape_ptr, const itype* __restrict__ out_sh, const dtype* dense_channel_buffer,
  const itype* __restrict__ filter_ids_kd, const itype* __restrict__ input_ids_kd, dtype* input_grads, dtype* filter_grads, int batch, int block_id_start, 
  int block_id_end, int filter_start, int filter_end)
{
  //1. define variables needed for convolution (overhead)
  const int block_start = block_id_start;
  const int block_end = block_id_end;
  const int filter_weight_start = filter_start;
  const int filter_weights_end = filter_end;
  const int ch_filter_weight_count = filter_weights_end - filter_weight_start;
  const int input_data_count = block_end - block_start;
  const int operation_count = ch_filter_weight_count * input_data_count;
  //load data to registers
  itype filter_shape[data_dimension];
  itype input_shape[data_dimension];
  for(int i = 0; i < data_dimension; ++i){
    filter_shape[i] = filter_shape_ptr[i];
  }
  for(int i = 0; i < data_dimension; ++i){
    input_shape[i] = in_shape_ptr[i];
  }
  itype out_id[data_dimension - 2];
  //2. perform convolution with kd indices
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      const int fid = y;
      const int did = x;
      const itype* data_index_kd = &input_ids_kd[(did + block_start) * (data_dimension - 2)];
      const itype* filter_index_kd = &filter_ids_kd[(fid + filter_weight_start) * (data_dimension - 2)];
      const int block_data_id1d = did + block_start;
      const int filter_data_id1d = fid + filter_weight_start;
      const dtype fv = filter_weights[filter_data_id1d];
      const dtype iv = in_block_vals[block_data_id1d];
      itype acc_id = 0;
      bool is_valid = true;
      int mul = 1;
      for(int i = 0; i < data_dimension - 2; ++i){
        out_id[i] = data_index_kd[i] - filter_index_kd[i] + (filter_shape[i] - 1) / 2;
        if(out_id[i] < 0 || out_id[i] >= input_shape[i + 1]){
          is_valid = false;
          break;
        }
        acc_id += out_id[i] * mul;
        mul = mul * input_shape[i + 1];
      }
      if(is_valid){
        dtype output_grad = dense_channel_buffer[acc_id];
        dtype df = iv * output_grad;
        dtype di = fv * output_grad;
        if(df != 0)
          atomicAdd(&(filter_grads[filter_data_id1d]), df);
        if(di != 0)
          atomicAdd(&(input_grads[block_data_id1d]), di);
      }
    }
  }
}


template <typename dtype, typename itype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
fill_channel_buffer(CudaLaunchConfig config, const itype* in_op_idkd, const dtype* in_op_val, const itype* out_shape, dtype* out_buffer, int data_start){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    itype acc_id = 0;
    bool is_valid = true;
    int mul = 1;
    const itype* out_id = &in_op_idkd[(x + data_start) * (data_dimension - 2)];
    for(int i = 0; i < data_dimension - 2; ++i){
      acc_id += out_id[i] * mul;
      mul = mul * out_shape[i + 1]; 
    } 
    out_buffer[acc_id] = in_op_val[x + data_start];
  }
}

template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseConvBackPropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, float max_density, const std::string& filter_type) const {
  const Tensor  *in_indices, *in_values, *in_shape, *in_block_ptr_t, *out_indices, *out_values, *out_shape, *out_block_ptr_t, 
                *data_count_t, *filter_indices, *filter_values, *filter_shape, *filter_channel_mapping_t, *gradients;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
  OP_REQUIRES_OK(context, context->input("out_values", &out_values));
  OP_REQUIRES_OK(context, context->input("out_shape", &out_shape));
  OP_REQUIRES_OK(context, context->input("out_block_channel_mapping", &out_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  OP_REQUIRES_OK(context, context->input("filter_channel_mapping", &filter_channel_mapping_t));
  OP_REQUIRES_OK(context, context->input("grads", &gradients));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto i_mapping = in_block_ptr_t->flat<int>();
  const int* input_block_mapping = i_mapping.data();
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_mapping = out_block_ptr_t->flat<int>();
  const int* output_block_mapping = o_mapping.data();
  auto f_sh = filter_shape->flat<IndiceT>();
  auto f_ind = filter_indices->flat<IndiceT>();
  auto f_val = filter_values->flat<T>(); 
  auto f_mapping = filter_channel_mapping_t->flat<int>();
  auto grads = gradients->flat<T>(); 
  int data_entry_count, filter_weight_count, output_entry_count;
  cudaMemcpy(&data_entry_count, i_mapping.data() + i_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&output_entry_count, o_mapping.data() + o_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&filter_weight_count, f_mapping.data() + f_mapping.dimension(0) - 1, sizeof(int), cudaMemcpyDeviceToHost);
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  std::vector<IndiceT> cpu_in_shape(data_dimension);
  cudaMemcpy(&cpu_in_shape[0], i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  //LOG(INFO) << "conv bp" << data_entry_count << " : " << cpu_in_shape[0] << " " << cpu_in_shape[1] << " " << cpu_in_shape[2] << " " << cpu_in_shape[3] << " " << cpu_in_shape[4] << std::endl;
  IndiceT channel_dense_size = 1;
  int max_dim = 0;
  for(size_t i = 0; i < data_dimension - 2; ++i){
    channel_dense_size = (channel_dense_size * cpu_in_shape[i + 1]); //x,y,z, ... (no channel and no batch) rounded up
  }
  const IndiceT batch_count = cpu_in_shape[0];
  const IndiceT in_channel_count = cpu_in_shape[data_dimension - 1];
  const IndiceT tensor_dense_size = channel_dense_size * batch_count * in_channel_count;

  const IndiceT *in_block_ids = i_ind.data();
  const T *in_block_vals = i_val.data();
  std::vector<int> cpu_input_block_mapping(batch_count * in_channel_count + 1);
  cudaMemcpy(&cpu_input_block_mapping[0], input_block_mapping, (batch_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<int> cpu_output_block_mapping(batch_count * out_channel_count + 1);
  cudaMemcpy(&cpu_output_block_mapping[0], output_block_mapping, (batch_count * out_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  Tensor fcm_tensor, fss_tensor, fse_tensor, fsw_tensor, fsi_tensor;
  const int* filter_channel_mapping = filter_channel_mapping_t->flat<int>().data();
  const T* filter_sorted_weights = f_val.data();
  const IndiceT* filter_sorted_ind_1d = f_ind.data();
  std::vector<int> cpu_filter_channel_mapping(out_channel_count * in_channel_count + 1);
  cudaMemcpy(&cpu_filter_channel_mapping[0], filter_channel_mapping, (out_channel_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  /////
  //1. Compute out shape
  //TODO
  Tensor out_sh_tensor;
  IndiceT *out_sh = 0;
  allocate_tensor(context, out_sh_tensor, &out_sh,  data_dimension);
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //2. decompress 1d to kd indice in temporary buffer
  //TODO: is a better approach possible? (decompression in kernels)
  Tensor in_id_kd_buffer, filter_id_kd_buffer, out_id_kd_buffer;
  IndiceT *in_id_kd_ptr = 0, *filter_id_kd_ptr = 0, *out_id_kd_ptr = 0;
  allocate_tensor(context, in_id_kd_buffer, &in_id_kd_ptr, (data_dimension - 2) * data_entry_count);
  allocate_tensor(context, filter_id_kd_buffer, &filter_id_kd_ptr, (data_dimension - 2) * filter_weight_count);
  allocate_tensor(context, out_id_kd_buffer, &out_id_kd_ptr, (data_dimension - 2) * output_entry_count);
  CudaLaunchConfig config_dec = GetCudaLaunchConfig(data_entry_count + filter_weight_count + output_entry_count, d);
  decompress_1d_to_kd<IndiceT, data_dimension><<<config_dec.block_count, config_dec.thread_per_block, 0, d.stream()>>>(config_dec, in_block_ids, filter_sorted_ind_1d, o_ind.data(), i_sh.data(), f_sh.data(), o_sh.data(), in_id_kd_ptr, filter_id_kd_ptr, out_id_kd_ptr, 0, data_entry_count, 0, filter_weight_count, 0, output_entry_count);
  
  cudaStreamSynchronize(d.stream());

  /////
  //3. Perform convolution; upper bound of output shape is known by max_density
  CudaLaunchConfig config_buffer = GetCudaLaunchConfig(channel_dense_size, d);
  CudaLaunchConfig config_rbuffer = GetCudaLaunchConfig(channel_dense_size * max_density, d);
  Tensor channel_offset_tensor, result_block_count_tensor, result_dense_count_tensor;
  
  const IndiceT result_count = ceil(tensor_dense_size * max_density);
  const int max_channel_count = floor(result_count / double(out_channel_count * batch_count));
  int *result_block_count, *result_dense_count;
  allocate_tensor(context, result_block_count_tensor, &result_block_count, 1);
  allocate_tensor(context, result_dense_count_tensor, &result_dense_count, 1);
  cudaMemset(result_block_count, 0, sizeof(int)); //stores the dense result of the computed output channel in buffer
  Tensor channel_buffer_tensor;
  T* channel_buffer = 0;
  allocate_tensor(context, channel_buffer_tensor, &channel_buffer, channel_dense_size);
  Tensor *input_grads = NULL, *filter_grads = NULL;
  TensorShape out_i_shape = {(IndiceT) i_val.dimension(0)};
  TensorShape out_f_shape = {(IndiceT) f_val.dimension(0)};
  OP_REQUIRES_OK(context, context->allocate_output("input_grads", out_i_shape, &input_grads));
  OP_REQUIRES_OK(context, context->allocate_output("filter_grads", out_f_shape, &filter_grads));
  auto in_grads = input_grads->flat<T>();
  auto f_grads = filter_grads->flat<T>();
  cudaMemset(in_grads.data(), 0, i_val.dimension(0) * sizeof(T));
  cudaMemset(f_grads.data(), 0, f_val.dimension(0) * sizeof(T));
  for(int i = 0; i < batch_count; ++i){
    for(int j = 0; j < out_channel_count; ++j){
      cudaStreamSynchronize(d.stream());
      cudaMemset(channel_buffer, 0, channel_dense_size * sizeof(T)); //stores the dense result of the computed output channel in buffer
      cudaStreamSynchronize(d.stream());
      int out_c_start = cpu_output_block_mapping[i * out_channel_count + j];
      int out_c_end = cpu_output_block_mapping[i * out_channel_count + j + 1]; //all blocks for batch
      if(out_c_end - out_c_start == 0) continue;
      CudaLaunchConfig config_obuff = GetCudaLaunchConfig(out_c_end - out_c_start, d);
      fill_channel_buffer<T, IndiceT, data_dimension><<<config_obuff.block_count, config_obuff.thread_per_block,0, d.stream()>>>(config_obuff, out_id_kd_ptr, grads.data(), o_sh.data(), channel_buffer, out_c_start);
      for(int k = 0; k < in_channel_count; ++k){
        int block_start_ = cpu_input_block_mapping[i * in_channel_count + k];
        int block_end_ = cpu_input_block_mapping[i * in_channel_count + k + 1]; //all blocks for batch
        int filter_start = cpu_filter_channel_mapping[j * in_channel_count + k];
        int filter_end = cpu_filter_channel_mapping[j * in_channel_count + k + 1];
        int data_block_count = block_end_ - block_start_;
        int op_count = (filter_end - filter_start) * data_block_count;
        if(op_count <= 0) continue;
        // int filter_weight_count, int hypercube_size_, int filter_max_dim, int batch, int block_id_start, int block_id_end, int filter_start, int filter_end
        Cuda2DLaunchConfig config_conv_ = GetCuda2DLaunchConfig(data_block_count, filter_end - filter_start, d);
        gmSparseDirectConvBackProp<T, IndiceT, data_dimension><<<config_conv_.block_count, config_conv_.thread_per_block, 0, d.stream()>>>(config_conv_,
                 in_block_vals, i_sh.data(),
                 filter_sorted_weights, f_sh.data(),
                 out_sh, channel_buffer, filter_id_kd_ptr, in_id_kd_ptr, in_grads.data(), f_grads.data(),
                 i, block_start_, block_end_, filter_start, filter_end);
      }
    }
  }
}

}  // end namespace functor

// Instantiate the GPU implementation for float.
//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseConvFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseConvBackPropFunctor<GPUDevice, type, indice_type, dim>;
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
