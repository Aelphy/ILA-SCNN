#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_approx_conv_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {

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
hashSparseDirectConv(Cuda2DLaunchConfig config, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_shape_ptr, 
  const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape_ptr, const itype* __restrict__ out_sh,
  const itype* __restrict__ filter_ids_kd, const itype* __restrict__ input_ids_kd,
  const itype* __restrict__ hash_table, const itype* __restrict__ hash_values, HashConfig hc, dtype* out_vals,
  int batch, int channel, int block_id_start, int block_id_end, int filter_start, int filter_end)
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
  itype out_id_[data_dimension];
  out_id_[0] = batch;
  out_id_[data_dimension - 1] = channel;
  itype* out_id = &out_id_[1];
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
      bool is_valid = true;
      for(int i = 0; i < data_dimension - 2; ++i){
        out_id[i] = data_index_kd[i] - filter_index_kd[i] + (filter_shape[i] - 1) / 2;
        if(out_id[i] < 0 || out_id[i] >= input_shape[i + 1]){
          is_valid = false;
          break;
        }
      }
      if(is_valid){
        itype out_id_1d = -1;
        index_KDto1D_<itype, data_dimension>(out_id_, out_sh, &out_id_1d);
        itype hash_result_id;
        querry_hash_table(&hash_result_id, hash_table, &out_id_1d, hc); 
        if(hash_result_id >= 0){
          int cid = hash_values[hash_result_id];
          atomicAdd(&(out_vals[cid]), out_v);
        }        
      }
    }
  }
}


namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseApproxConvFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, float max_density, const std::string& filter_type) const {
  clock_t t;
  const Tensor *in_indices, *in_values, *in_shape, *in_block_ptr_t, *data_count_t, *filter_indices, *filter_values, *filter_shape, *filter_channel_mapping_t, *out_indices, *out_shape;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  OP_REQUIRES_OK(context, context->input("filter_channel_mapping", &filter_channel_mapping_t));
  OP_REQUIRES_OK(context, context->input("out_indices", &out_indices));
  OP_REQUIRES_OK(context, context->input("out_shape", &out_shape));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto f_sh = filter_shape->flat<IndiceT>();
  auto f_ind = filter_indices->flat<IndiceT>();
  auto f_val = filter_values->flat<T>(); 
  auto i_mapping = in_block_ptr_t->flat<int>();
  auto f_mapping = filter_channel_mapping_t->flat<int>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_sh = out_shape->flat<IndiceT>();
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
  const int tensor_dense_size = channel_dense_size * cpu_in_shape[0] * cpu_in_shape[data_dimension - 1];
  const int batch_count = cpu_in_shape[0];
  const int in_channel_count = cpu_in_shape[data_dimension - 1];

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
  HashConfig hc;
  Tensor hash_table, hash_values;
  initialize_table<DeviceT, IndiceT, IndiceT>(context, d, hash_table, hash_values, o_ind.data(), o_ind.data(), (IndiceT) o_ind.dimension(0), hc);
  auto hashv = (IndiceT*) hash_values.flat<int8>().data();
  auto hasht = (IndiceT*) hash_table.flat<int8>().data();

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
  Tensor *out_values = NULL;
  TensorShape out_val_shape = {(IndiceT) i_ind.dimension(0)};
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  auto o_val = out_values->flat<T>();
  cudaMemset(o_val.data(), 0, o_val.dimension(0) * sizeof(T));

  for(int i = 0; i < batch_count; ++i){
    for(int j = 0; j < out_channel_count; ++j){
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
        hashSparseDirectConv<T, IndiceT, data_dimension><<<config_conv_.block_count, config_conv_.thread_per_block, 0, d.stream()>>>(config_conv_,
                 in_block_vals, i_sh.data(),
                 filter_sorted_weights, f_sh.data(),
                 o_sh.data(), filter_id_kd_ptr, in_id_kd_ptr,
                 hasht, hashv, hc, o_val.data(),
                 i, j, block_start_, block_end_, filter_start, filter_end);
      }   
    }   
  }
  cudaStreamSynchronize(d.stream());
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
hashSparseDirectConvBackProp(Cuda2DLaunchConfig config, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_shape_ptr, 
  const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape_ptr, const itype* __restrict__ out_sh, const dtype* gradients, 
  const itype* __restrict__ filter_ids_kd, const itype* __restrict__ input_ids_kd, 
  const itype* __restrict__ hash_table, const itype* __restrict__ hash_values, HashConfig hc,
  dtype* input_grads, dtype* filter_grads, int batch, int channel, int block_id_start, 
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
  itype out_id_[data_dimension];
  out_id_[0] = batch;
  out_id_[data_dimension - 1] = channel;
  itype* out_id = &out_id_[1];
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
        itype out_id_1d = -1;
        index_KDto1D_<itype, data_dimension>(out_id_, out_sh, &out_id_1d);
        itype hash_result_id;
        querry_hash_table(&hash_result_id, hash_table, &out_id_1d, hc); 
        if(hash_result_id >= 0){
          int cid = hash_values[hash_result_id];
          dtype output_grad = gradients[cid];
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
}


template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseApproxConvBackPropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, float max_density, const std::string& filter_type) const {
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
  IndiceT channel_dense_size = 1;
  int max_dim = 0;
  for(size_t i = 0; i < data_dimension - 2; ++i){
    channel_dense_size = (channel_dense_size * cpu_in_shape[i + 1]); //x,y,z, ... (no channel and no batch) rounded up
  }
  const int tensor_dense_size = channel_dense_size * cpu_in_shape[0] * cpu_in_shape[data_dimension - 1];
  const int batch_count = cpu_in_shape[0];
  const int in_channel_count = cpu_in_shape[data_dimension - 1];

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
  //1. Compute hash table
  HashConfig hc;
  Tensor hash_table, hash_values;
  initialize_table<DeviceT, IndiceT, IndiceT>(context, d, hash_table, hash_values, i_ind.data(), i_ind.data(), (IndiceT) i_ind.dimension(0), hc);
  auto hashv = (IndiceT*) hash_values.flat<int8>().data();
  auto hasht = (IndiceT*) hash_table.flat<int8>().data();

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
  
  Tensor channel_buffer_tensor;
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
        hashSparseDirectConvBackProp<T, IndiceT, data_dimension><<<config_conv_.block_count, config_conv_.thread_per_block, 0, d.stream()>>>(config_conv_,
                 in_block_vals, i_sh.data(),
                 filter_sorted_weights, f_sh.data(),
                 o_sh.data(), grads.data(), filter_id_kd_ptr, in_id_kd_ptr, hasht, hashv, hc, in_grads.data(), f_grads.data(),
                 i, j, block_start_, block_end_, filter_start, filter_end);
      }
    }
  }
  cudaStreamSynchronize(d.stream());
}

}  // end namespace functor

// Instantiate the GPU implementation for float.
//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseApproxConvFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseApproxConvBackPropFunctor<GPUDevice, type, indice_type, dim>;
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
