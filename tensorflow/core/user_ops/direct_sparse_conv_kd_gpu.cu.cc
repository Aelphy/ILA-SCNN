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
map_channel_buffer_1d_to_kd(dtype in_id, const dtype* __restrict__ out_shape_ptr, dtype* out_index)
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
decompress_1d_to_kd(CudaLaunchConfig config, const dtype* in_data_id1d, const dtype* in_filter_id1d, const dtype* in_block_id1d, const dtype* in_shape_ptr, dtype* out_data_idkd, dtype* out_filter_idkd, dtype* out_block_idkd, int data_start, int data_end, int filter_start, int filter_end, int block_id_start, int block_id_end)
{
  int data_size = data_end - data_start;
  int filter_size = filter_end - filter_start;
  int block_id_size = block_id_end - block_id_start;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if(x < 0){  //x might overflow when testing extreme case
      break;
    }
    if(x < data_size){
      int id = x;
      index_1DtoKD_reduced<dtype, data_dimension>(0, in_data_id1d[id + data_start], in_shape_ptr, &out_data_idkd[(data_dimension - 2) * id]);
    } else if(x < data_size + filter_size){
      int id = x - data_size;
      index_1DtoKD_reduced<dtype, data_dimension>(0, in_filter_id1d[id + filter_start], in_shape_ptr, &out_filter_idkd[(data_dimension - 2) * id]);
    } else {
      int id = x - data_size - filter_size;
      decompress_block_id<dtype, data_dimension>(in_block_id1d[id + block_id_start], in_shape_ptr, &out_block_idkd[(data_dimension - 2) * id], true); 
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

template <typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_256_THREADS_PER_BLOCK, MIN_8_BLOCKS_PER_MP)
kdSparseDirectConv(CudaLaunchConfig config, const itype* __restrict__ in_block_ids, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_block_pointer, 
  const itype* __restrict__ in_block_pointer_id, const itype* __restrict__ in_shape_ptr, const int* input_block_mapping, 
  const itype* __restrict__ filter_ind_1d, const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape_ptr, const int* filter_channel_mapping,
  const itype* __restrict__ out_sh, dtype* dense_channel_buffer,
  const itype* __restrict__ filter_ids_kd, const itype* __restrict__ input_ids_kd, const itype* __restrict__ block_ids_kd,
  int data_entry_count, int filter_weight_count, int hypercube_size_, int filter_max_dim, int batch, int in_channel_count, int out_channel)
{
  //1. define variables needed for convolution (overhead)
  const int sm_size = 768;
  __shared__ dtype accumulated_result[sm_size]; //TODO: dynamic allocation
  int filter_start, filter_end, block_start_;
  conv_init_parameters(&filter_start, &filter_end, &block_start_, blockIdx.x, input_block_mapping, filter_channel_mapping, accumulated_result, batch, in_channel_count, out_channel);
  for(int x = threadIdx.x; x < sm_size; x += blockDim.x){
    accumulated_result[x] = 0;
  }
  const int block_id = block_start_;
  const int block_start = in_block_pointer[block_id];
  const int block_end = in_block_pointer[block_id + 1];
  const int filter_weight_start = filter_start;
  const int filter_weights_end = filter_end;
  const int ch_filter_weight_count = filter_weights_end - filter_weight_start;
  const int input_data_count = block_end - block_start;
  const int operation_count = ch_filter_weight_count * input_data_count;
  //load data to registers
  itype filter_shape[data_dimension];
  for(int i = 0; i < data_dimension; ++i){
    filter_shape[i] = filter_shape_ptr[i];
  }
  itype block_start_array[data_dimension - 2];
  for(int i = 0; i < data_dimension - 2; ++i){
   block_start_array[i] = block_ids_kd[(data_dimension - 2) * block_id + i];
  }
  int hypercube_size = hypercube_size_;
  __syncthreads();
  //2. perform convolution with kd indices
  for(int x = threadIdx.x; x < operation_count; x += blockDim.x){
    const int fid = x % ch_filter_weight_count;
    const int did = (x - fid) /  ch_filter_weight_count;
    const itype* data_index_kd = &input_ids_kd[(did + block_start) * (data_dimension - 2)];
    const itype* filter_index_kd = &filter_ids_kd[(fid + filter_weight_start) * (data_dimension - 2)];
    const int block_data_id1d = did + block_start;
    const int filter_data_id1d = fid + filter_weight_start;
    const dtype fv = filter_weights[filter_data_id1d];
    const dtype iv = in_block_vals[block_data_id1d];
    const dtype out_v = fv * iv;
    itype acc_id;
    map_index_kd_to_shared_id<itype, data_dimension>(data_index_kd, filter_index_kd, block_start_array, filter_shape, &acc_id, hypercube_size);
    atomicAdd(&(accumulated_result[acc_id]), out_v);
  }
  __syncthreads();
  //3. check if entries are valid (inside tensor shape) and write valid entries to global memory buffer
  for(int x = threadIdx.x; x < pow(hypercube_size + filter_max_dim - 1, data_dimension - 2); x += blockDim.x){
    itype local_id = x;
    if(accumulated_result[local_id] == 0) continue; //TODO: thread divergence: find better approach with less idle threads
    itype global_acc_id;
    map_shared_to_channel_buffer<itype, data_dimension>(local_id, block_start_array, filter_shape, in_shape_ptr, &global_acc_id, hypercube_size);
    if(global_acc_id >= 0){ //invalid ids are set to 0
      atomicAdd(&(dense_channel_buffer[global_acc_id]), accumulated_result[local_id]);
    }
  }
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
write_conv_res2(CudaLaunchConfig config, const dtype* in_buffer, const itype* in_ids, itype* out_ind, dtype* out_val, const itype* out_shape_ptr, int out_entries, int channel, int batch, int* count, int* offset_)
{
  itype offset = *offset_;
  itype idkd[data_dimension];
  idkd[0] = batch;
  idkd[data_dimension - 1] = channel;
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    itype idx = in_ids[x];
    dtype outval = in_buffer[idx];
    if(outval == 0) continue;
    itype* idkd_r = &idkd[1];
    map_channel_buffer_1d_to_kd<itype, data_dimension>(idx, out_shape_ptr, idkd_r);
    itype out_id_1d;
    index_KDto1D_<itype, data_dimension>(idkd, out_shape_ptr, &out_id_1d);
    int out_x = atomicAdd(count, 1);
    out_val[out_x] = outval;
    out_ind[out_x] = out_id_1d;
  }
}

template <typename dtype, typename itype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
abs_values(CudaLaunchConfig config, const dtype* __restrict__ in_buffer, itype* out_buffer){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    out_buffer[x] = itype(abs(in_buffer[x] * 1024));
  }
}


template <typename dtype> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
gen_sorted_index(CudaLaunchConfig config, dtype* out_buffer){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count){
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    out_buffer[x] = x;
  }
}

template <typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
gmSparseDirectConv(Cuda2DLaunchConfig config, const dtype* __restrict__ in_block_vals, const itype* __restrict__ in_block_pointer,   const itype* __restrict__ in_block_pointer_id, const itype* __restrict__ in_shape_ptr, const int* input_block_mapping, 
  const dtype* __restrict__ filter_weights, const itype* __restrict__ filter_shape_ptr, const itype* __restrict__ out_sh, dtype* dense_channel_buffer,
  const itype* __restrict__ filter_ids_kd, const itype* __restrict__ input_ids_kd, int batch, int block_id_start, int block_id_end, int filter_start, int filter_end)
{
  //1. define variables needed for convolution (overhead)
  const int block_start = in_block_pointer[block_id_start];
  const int block_end = in_block_pointer[block_id_end];
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
void ApproxDirectSparseConvFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, float max_density) const {
  clock_t t_total = clock();
  const Tensor *in_indices, *in_values, *in_shape, *in_block_ptr_t, *in_block_ptr_ids_t, *data_count_t, *filter_indices, *filter_values, *filter_shape, *filter_channel_mapping_t;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("in_block_ptr", &in_block_ptr_t));
  OP_REQUIRES_OK(context, context->input("in_block_ptr_ids", &in_block_ptr_ids_t));
  OP_REQUIRES_OK(context, context->input("in_data_count", &data_count_t));
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  OP_REQUIRES_OK(context, context->input("filter_channel_mapping", &filter_channel_mapping_t));
  const DeviceT d = context->eigen_device<DeviceT>();
  int device_version = d.majorDeviceVersion();
  if(device_version < 6){
    LOG(WARNING) << "compute capability to low; requires 6.0 or higher for fast sparse convolution" << std::endl; //atomics for shared memory, max blocks per sm, etc...
  }
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->flat<IndiceT>();
  auto i_val = in_values->flat<T>();
  auto f_sh = filter_shape->flat<IndiceT>();
  auto f_ind = filter_indices->flat<IndiceT>();
  auto f_val = filter_values->flat<T>(); 
  const int data_entry_count = i_ind.dimension(0);
  const int filter_weight_count = f_ind.dimension(0);
  const int smpb = d.sharedMemPerBlock();
  const int mtpb = d.maxCudaThreadsPerBlock();
  const int max_blocks_per_sm = 64; //TODO: check
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  std::vector<IndiceT> cpu_in_shape(data_dimension);
  cudaMemcpy(&cpu_in_shape[0], i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  IndiceT channel_dense_size = 1;
  int max_dim = 0;
  for(size_t i = 0; i < data_dimension - 2; ++i){
    channel_dense_size = (channel_dense_size * cpu_in_shape[i + 1]); //x,y,z, ... (no channel and no batch) rounded up
    max_dim = max(max_dim, (int) cpu_in_shape[i + 1]);
  }
  const int tensor_dense_size = channel_dense_size * cpu_in_shape[0] * cpu_in_shape[data_dimension - 1];
  const int batch_count = cpu_in_shape[0];
  const int in_channel_count = cpu_in_shape[data_dimension - 1];
  const int filter_size = 3; //TODO
  const int hypercube_size_ = floor(pow(float(smpb / sizeof(T) / 16), 1. / (data_dimension - 2))) - (filter_size - 1); //compute block size: assumptions: i) all dimensions have the same size (not necessarly true); ii) two blocks per sm
  const int hypercube_size = min(hypercube_size_, max_dim); //TODO: remove after debugging
  const int dense_block_count = pow(hypercube_size, data_dimension - 2);
  const int dense_filter_block_count = pow(filter_size, data_dimension - 2); //TODO

  std::stringstream dout_s;
  //indices must! be sorted
  clock_t t;

  //preprocessing step (1) has to be performed only for one layer in the neural network! Also step (2) can be precomputed and shouldn't affect runtime of nn
  
  /////
  //1. Convert Coordinate List Format to sparse block format and compress k dimensional indices to 1d
  t = clock();
  if(hypercube_size <= 0) return; //TODO: THROW ERROR
  const IndiceT *in_block_ids = i_ind.data();
  const IndiceT *in_block_pointer = in_block_ptr_t->flat<IndiceT>().data();
  const IndiceT* in_block_pointer_ids = in_block_ptr_ids_t->flat<IndiceT>().data();
  const T *in_block_vals = i_val.data();
  int block_count = in_block_ptr_t->flat<IndiceT>().dimension(0) - 1;

  LOG(INFO) << "Edge length: " << hypercube_size << " Shared memory per block: " << smpb << " sizeof T " << sizeof(T) << std::endl;
  dout_s << "t1: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  LOG(INFO) << dout_s.str(); dout_s.str("");

  /////
  //2. prepare filter: directly manipulate 1D keys instead of kD indices and flip filter weights to be applicable for direct convolution and sort filter w.r.t. output and input channels
  t = clock();
  Tensor fcm_tensor, fss_tensor, fse_tensor, fsw_tensor, fsi_tensor;
  const int* filter_channel_mapping = filter_channel_mapping_t->flat<int>().data();
  const T* filter_sorted_weights = f_val.data();
  const IndiceT* filter_sorted_ind_1d = f_ind.data();
  std::vector<int> cpu_filter_channel_mapping(out_channel_count * in_channel_count + 1);
  cudaMemcpy(&cpu_filter_channel_mapping[0], filter_channel_mapping, (out_channel_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  dout_s << "t2: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  //LOG(INFO) << dout_s.str(); dout_s.str("");

  /////
  //3. Get mapping of blocks / channels (input) and input channels / output channels (filter)
  t = clock();
  Tensor input_block_mapping_tensor;
  int* input_block_mapping = 0;
  allocate_tensor(context, input_block_mapping_tensor, &input_block_mapping,  (batch_count * in_channel_count + 1));
  CudaLaunchConfig ib_config = GetCudaLaunchConfig(max(block_count, batch_count * in_channel_count + 1), d);
  compute_input_block_index<IndiceT, data_dimension><<<ib_config.block_count, ib_config.thread_per_block, 0, d.stream()>>>(ib_config, /*in_block_ids,*/ 
    in_block_pointer_ids, input_block_mapping, i_sh.data(), block_count, batch_count, in_channel_count);
  std::vector<int> cpu_input_block_mapping(batch_count * in_channel_count + 1);
  cudaMemcpy(&cpu_input_block_mapping[0], input_block_mapping, (batch_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
  std::vector<IndiceT> cpu_block_pointers(block_count + 1);
  cudaMemcpy(&cpu_block_pointers[0], in_block_pointer, (block_count + 1) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  dout_s << "t3: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  
  /////
  //4. Compute out shape
  //TODO
  Tensor out_sh_tensor;
  IndiceT *out_sh = 0;
  allocate_tensor(context, out_sh_tensor, &out_sh,  data_dimension);
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //5. decompress 1d to kd indice in temporary buffer
  //TODO: is a better approach possible? (decompression in kernels)
  t = clock(); 
  Tensor in_id_kd_buffer, block_id_kd_buffer, filter_id_kd_buffer;
  IndiceT *in_id_kd_ptr = 0, *block_id_kd_ptr = 0, *filter_id_kd_ptr = 0;
  allocate_tensor(context, in_id_kd_buffer, &in_id_kd_ptr, (data_dimension - 2) * data_entry_count);
  allocate_tensor(context, filter_id_kd_buffer, &filter_id_kd_ptr, (data_dimension - 2) * filter_weight_count);
  allocate_tensor(context, block_id_kd_buffer, &block_id_kd_ptr, (data_dimension - 2) * block_count);
  CudaLaunchConfig config_dec = GetCudaLaunchConfig(data_entry_count + filter_weight_count + block_count, d);
  decompress_1d_to_kd<IndiceT, data_dimension><<<config_dec.block_count, config_dec.thread_per_block, 0, d.stream()>>>(config_dec, in_block_ids, filter_sorted_ind_1d, in_block_pointer_ids, i_sh.data(), in_id_kd_ptr, filter_id_kd_ptr, block_id_kd_ptr, 0, data_entry_count, 0, filter_weight_count, 0, block_count);
  
  cudaDeviceSynchronize(); 
  dout_s << "t6: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  /////
  //6. Perform convolution; upper bound of output shape is known by max_density
  t = clock();
  CudaLaunchConfig config_buffer = GetCudaLaunchConfig(channel_dense_size, d);
  CudaLaunchConfig config_rbuffer = GetCudaLaunchConfig(channel_dense_size * max_density, d);
  Tensor channel_offset_tensor, result_block_count_tensor;
  
  int result_count = ceil(tensor_dense_size / max_density);
  int* result_block_count;
  allocate_tensor(context, result_block_count_tensor, &result_block_count, 1);
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
  TensorShape out_count_shape = {(IndiceT) 1};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  OP_REQUIRES_OK(context, context->allocate_output("out_data_count", out_count_shape, &data_count));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->matrix<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto data_offset = data_count->flat<int>();
  gen_sorted_index<<<config_buffer.block_count, config_buffer.thread_per_block, 0, d.stream()>>>(config_buffer, in_channel_ids_buffer);
  gen_sorted_index<<<config_buffer.block_count, config_buffer.thread_per_block, 0, d.stream()>>>(config_buffer, sorted_channel_ids_buffer);
  for(int i = 0; i < batch_count; ++i){
    for(int j = 0; j < out_channel_count; ++j){
      cudaStreamSynchronize(d.stream());
      cudaMemset(channel_buffer, 0, channel_dense_size * sizeof(T)); //stores the dense result of the computed output channel in buffer
      for(int k = 0; k < in_channel_count; ++k){
        int block_start_ = cpu_input_block_mapping[i * in_channel_count + k];
        int block_end_ = cpu_input_block_mapping[i * in_channel_count + k + 1]; //all blocks for batch
        int filter_start = cpu_filter_channel_mapping[j * in_channel_count + k];
        int filter_end = cpu_filter_channel_mapping[j * in_channel_count + k + 1];
        int data_block_count = cpu_block_pointers[block_end_] - cpu_block_pointers[block_start_];
        int op_count = (filter_end - filter_start) * data_block_count;
        if(op_count <= 0) continue;
        // int filter_weight_count, int hypercube_size_, int filter_max_dim, int batch, int block_id_start, int block_id_end, int filter_start, int filter_end
        Cuda2DLaunchConfig config_conv_ = GetCuda2DLaunchConfig(data_block_count, filter_end - filter_start, d);
        gmSparseDirectConv<T, IndiceT, data_dimension><<<config_conv_.block_count, config_conv_.thread_per_block, 0, d.stream()>>>(config_conv_,
                 in_block_vals, in_block_pointer, in_block_pointer_ids, i_sh.data(), input_block_mapping,
                 filter_sorted_weights, f_sh.data(),
                 out_sh, channel_buffer, filter_id_kd_ptr, in_id_kd_ptr,
                 i, block_start_, block_end_, filter_start, filter_end);
      }
      cudaStreamSynchronize(d.stream());
      abs_values<<<config_buffer.block_count, config_buffer.thread_per_block, 0, d.stream()>>>(config_buffer, channel_buffer, abs_channel_buffer);
      cudaStreamSynchronize(d.stream());
      compute_sort(context, d, abs_channel_buffer, tmp_channel_buffer, in_channel_ids_buffer, sorted_channel_ids_buffer, channel_dense_size); //TODO: sort decreasing
      cudaStreamSynchronize(d.stream());
      write_conv_res2<T, IndiceT, data_dimension><<<config_rbuffer.block_count, config_rbuffer.thread_per_block, 0, d.stream()>>>(config_rbuffer, channel_buffer, 
          sorted_channel_ids_buffer, o_ind.data(), o_val.data(), i_sh.data(), channel_dense_size * max_density, j, i, result_block_count, data_offset.data());
      cudaMemcpy(data_offset.data(), result_block_count, sizeof(IndiceT), cudaMemcpyDeviceToDevice);
    }
  }
  //TODO: write output block ids
  cudaMemcpy(o_sh.data(), out_sh, data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  dout_s << "t7: " << float(clock() - t)/CLOCKS_PER_SEC << std::endl;

  cudaDeviceSynchronize();
  dout_s << "t_total: " << float(clock() - t_total)/CLOCKS_PER_SEC << std::endl;

  LOG(INFO) << dout_s.str();
}
}  // end namespace functor

// Instantiate the GPU implementation for float.
//template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, type, indice_type, dim>;
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
