#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "direct_sparse_conv_kd_gpu.h"
#include "tf_cudpp_bindings_gpu.h"

namespace tensorflow {

inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    //TODO: use tensorflow logging levels
    fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

//Compress [batch, x, y, ...] indices into a [1D] key while keeping the data sorted. Except for [channel], which is handled seperately.
template <typename dtype>
__global__ void index_KDto1D(CudaLaunchConfig config, const dtype* in_ptr, const dtype* in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, dtype* out_channels_ptr, const int dimension_count, const int entry_count){
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
__global__ void index_1DtoKD(CudaLaunchConfig config, const dtype* in_1D_ptr, const dtype* in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    const dtype* in_channels_ptr, dtype* out_ind_ptr, const int dimension_count, const int entry_count){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    //out_ind_ptr[x] = 0;
    dtype idx = x * dimension_count;
    //1. compressed 1d key, except channel
    dtype *fact = new dtype[dimension_count - 1];
    fact[dimension_count - 2] = 1;
    for(int i = dimension_count - 3; i >= 0; i = i - 1){
      fact[i] = fact[i + 1] * in_shape_ptr[i + 1];
    }
		dtype r = in_1D_ptr[x];
    for(int i = 0; i < dimension_count - 1; ++i){
      out_ind_ptr[idx + i] = r / fact[i];
      r = r % fact[i];
    }
		delete[] fact;
    //2. add channel
    out_ind_ptr[idx + dimension_count - 1] = in_channels_ptr[x];
  }
}

//mark unique elemets in an array with a $1$
template <typename dtype>
__global__ void unique_mask(CudaLaunchConfig config, const dtype* in_ptr, dtype* out_ptr){
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

//mark mark non-zero elemets in an array with a $1$
template <typename dtype, typename itype>
__global__ void non_zero_mask(CudaLaunchConfig config, const dtype* in_ptr, itype* out_ptr){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(x == 0){
      out_ptr[x] = 1;
    } else {
      if(in_ptr[x] != 0){
        out_ptr[x] = 1;
      } else {
        out_ptr[x] = 0;
      }
    }
  }
}

//copy obtain unique elemets from array
template <typename dtype, typename itype>
__global__ void unique_array(CudaLaunchConfig config, const dtype* in_id_ptr, const itype* unique_masked_ptr, 
              const itype* unique_count, dtype* unique_ptr){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    if(unique_masked_ptr[x] == 1){
      unique_ptr[unique_count[x] - 1] = in_id_ptr[x];
    }
  }
}

template <typename dtype>
__global__ void prepare_filter_weights_(CudaLaunchConfig config, const dtype* f_id_ptr, const dtype* f_sh_ptr, const dtype* in_sh_ptr,
    dtype* out_id_ptr, const int dimension_count, const int data_entry_count, const int filter_entry_count){
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
      val = val + mul * (f_id_ptr[idx] - offset);
      mul = mul * in_sh_ptr[i];
    }   
    //const dtype channel = in_ptr[(x + 1)  * dimension_count - 1]; 
    out_id_ptr[x] = val;
  }
}

template<typename dtype>
__device__ void index_lookup(const dtype index, const dtype *data,  const dtype data_size, dtype* result_id){
  //binary search
  *result_id = 0;  
  dtype upper = data_size - 1;
  dtype lower = 0;
  while(lower <= upper){
    dtype center = lower + (upper - lower) / 2;
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
  *result_id = -1;
}

template <typename dtype, typename itype>
__global__ void approxSparseDirectConv(Cuda2DLaunchConfig config, 
   const itype* i_ind, const dtype* i_val, const itype* i_sh, const itype* i_ind_1d, const itype* i_ch, //input tensors
   const itype* f_ind, const dtype* f_val, const itype* f_sh, const itype* f_ind_1d, //filter tensors
   const itype* r_ind, const itype reduced_count, //search structure
   dtype* out_conv_data, const itype* out_sh,
   const int data_entry_count, const int filter_weight_count, const int data_dimension){

  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) { //index of feature map
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) { //index of filter weights
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      //1. check valid indice and valid stride / padding:
      itype idx = data_dimension * x;
      itype idy = data_dimension * y;
      for(int i = 1; i < data_dimension - 1; ++i){
        itype id = i_ind[idx + i] + f_ind[idy + i - 1] - (f_sh[i - 1] - 1) / 2;
        if(id < 0 || id >= out_sh[i]) return;
        //TODO: stride and padding
      }
      //TODO: check channel filter/input
      
      //2. compute update indice
      itype lookup_id = i_ind_1d[x] + f_ind_1d[y];
      itype update_id = 0;
      index_lookup(lookup_id, r_ind, reduced_count, &update_id);
      //3. update indice
      if(update_id < 0) return; //id not found in search structure -> this is an error and should not occure //TODO throw error
      itype channel_offset = reduced_count * i_ch[x];
      const float update_val = f_val[y] * i_val[x];
      atomicAdd(&out_conv_data[update_id + channel_offset], update_val);
     
    }
  }
}

template<typename dtype>
void compute_scan(dtype* out, const dtype* in, const int count){
  CUDPPHandle cudppHandle; //compute scan with cudpp lib
  cudppCreate(&cudppHandle);
  auto config = getConfiguration<dtype>("add", "scan");
  CUDPPHandle scanplan = 0;
  CUDPPResult res = cudppPlan(cudppHandle, &scanplan, config, count, 1, 0);
  res = cudppScan(scanplan, out, in, count);
  if (CUDPP_SUCCESS != res){
      printf("Error in cudppScan()\n"); //TODO: use tensorflow logging
  }
  cudppDestroy(cudppHandle);
}

namespace functor {
template <typename DeviceT, typename T, typename IndiceT>
void ApproxDirectSparseConvFunctor<DeviceT, T, IndiceT>::operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, const IndiceT& filter_dim) const {
  const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
  Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL;
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

  //indices must! be sorted

  //output for debugging
  TensorShape out_ind_shape = {data_entry_count, data_dimension};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  auto o_ind = out_indices->matrix<IndiceT>();
  TensorShape out_val_shape = {data_entry_count};
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  auto o_val = out_values->flat<T>();
  TensorShape out_sh_shape = {data_entry_count};
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  auto o_sh = out_shape->flat<IndiceT>();

  /////
  //1. set channel to 0 and convert indices to 1D key
  // + define rule to work with more than one channel
  IndiceT *in_ind_1d = 0;
  checkCuda(cudaMalloc(&in_ind_1d, data_entry_count * sizeof(IndiceT)));
  IndiceT *in_ind_1d_channels = 0;
  checkCuda(cudaMalloc(&in_ind_1d_channels, data_entry_count * sizeof(IndiceT)));
  CudaLaunchConfig config_i1d = GetCudaLaunchConfig(data_entry_count, d);
  index_KDto1D<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d,
      i_ind.data(), i_sh.data(),  in_ind_1d, in_ind_1d_channels, data_dimension, data_entry_count);
  //cudaMemcpy(o_sh.data(), in_ind_1d, data_entry_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  //cudaMemcpy(o_sh.data(), in_ind_1d_channels, data_entry_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  //index_1DtoKD<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d,
  //    in_ind_1d, i_sh.data(),  in_ind_1d_channels, o_ind.data(), data_dimension, data_entry_count);
  
  
  /////
  //2. remove duplicates from data and apply stride/padding to obtain search structure
  IndiceT *unique_masked = 0;
  checkCuda(cudaMalloc(&unique_masked, data_entry_count * sizeof(IndiceT)));
  unique_mask<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d, in_ind_1d, unique_masked);
  //cudaMemcpy(o_sh.data(), unique_masked, data_entry_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  IndiceT *unique_count = 0;
  checkCuda(cudaMalloc(&unique_count, data_entry_count * sizeof(IndiceT)));
  compute_scan(unique_count, unique_masked, data_entry_count);
  cudaMemcpy(o_sh.data(), unique_count, data_entry_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  IndiceT reduced_count = -1;
  cudaMemcpy(&reduced_count, unique_count + data_entry_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  IndiceT* reduced_indices = 0;
  checkCuda(cudaMalloc(&reduced_indices, reduced_count * sizeof(IndiceT)));
  unique_array<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d, in_ind_1d, unique_masked, unique_count, reduced_indices);
  cudaMemcpy(o_sh.data(), reduced_indices, reduced_count* sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  //TODO: apply stride/padding
  //TODO: initialize search structure
  
  
  /////
  //3. prepare filter (to directly manipulate 1D keys instead of kD indices)
  IndiceT *filter_ind_1d = 0;
  checkCuda(cudaMalloc(&filter_ind_1d, filter_weight_count * sizeof(IndiceT)));
  CudaLaunchConfig config_f1d = GetCudaLaunchConfig(filter_weight_count, d);
  prepare_filter_weights_<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, 
    f_ind.data(), f_sh.data(), i_sh.data(), filter_ind_1d,  data_dimension, data_entry_count, filter_weight_count);
  cudaMemcpy(o_sh.data(), filter_ind_1d, filter_weight_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //4. compute out shape
  
  //TODO
  IndiceT *out_sh = 0;
  checkCuda(cudaMalloc(&out_sh, data_dimension * sizeof(IndiceT)));
  cudaMemcpy(out_sh, i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);

  /////
  //5. perform approximated convolution
  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  T* conv_res = 0;
  IndiceT conv_out_count = out_channel_count * reduced_count;
  //if(conv_out_count <= 0) return;
  checkCuda(cudaMalloc(&conv_res, conv_out_count * sizeof(T)));
  cudaMemset(conv_res, 0, conv_out_count * sizeof(T));
  Cuda2DLaunchConfig config_conv = GetCuda2DLaunchConfig(data_entry_count, filter_weight_count, d);
  approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
    i_ind.data(), i_val.data(), i_sh.data(), in_ind_1d, in_ind_1d_channels,
    f_ind.data(), f_val.data(), f_sh.data(), filter_ind_1d,
    reduced_indices, reduced_count, conv_res, out_sh,
    data_entry_count, filter_weight_count, data_dimension);

  /////
  //6. remove zero entries and convert from keys to indices
  IndiceT *non_zero_masked = 0;
  checkCuda(cudaMalloc(&non_zero_masked, conv_out_count * sizeof(IndiceT)));
  CudaLaunchConfig config_r1d = GetCudaLaunchConfig(conv_out_count, d);
  non_zero_mask<<<config_r1d.block_count, config_r1d.thread_per_block, 0, d.stream()>>>(config_r1d, conv_res, non_zero_masked);
  IndiceT *non_zero_count = 0;
  checkCuda(cudaMalloc(&non_zero_count, conv_out_count * sizeof(IndiceT)));
  //compute_scan(non_zero_count, non_zero_masked, conv_out_count);
  //IndiceT result_count = -1;
  //cudaMemcpy(&result_count, non_zero_count + conv_out_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  

  /*
  // Create an output tensor
  Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL;
  TensorShape out_ind_shape = {(Tindices) in_ind.dimension(0), (Tindices) in_ind.dimension(1)};
  TensorShape out_val_shape = {(Tindices) in_vals.dimension(0)};
  TensorShape out_sh_shape = {(Tindices) in_sh.dimension(0)};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  */

  //# free memory
  cudaFree(in_ind_1d);
  cudaFree(in_ind_1d_channels);
  cudaFree(unique_masked);
  cudaFree(unique_count);
  cudaFree(reduced_indices);
  cudaFree(filter_ind_1d);
  cudaFree(out_sh);
  cudaFree(conv_res);
  cudaFree(non_zero_masked);
  cudaFree(non_zero_count);
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
