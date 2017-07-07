#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "direct_sparse_conv_kd_gpu.h"
//#include "tf_cudpp_bindings_gpu.h"

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



template <typename dtype>
__global__ void approxSparseDirectConv(Cuda2DLaunchConfig config, const dtype* input_ptr,
                    dtype* output_ptr) {
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      atomicAdd(&output_ptr[x], 1);
    }
  }
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
  
  const int data_entry_count = i_ind.dimension(0);
  const int data_dimension = i_ind.dimension(1);
  //const int filter_weight_count = f_ind.dimension(0);

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
  //prescan<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(unique_count, unique_masked, data_entry_count);
  /*
  /////
  //3. prepare filter (to directly manipulate 1D keys instead of kD indices)
  
  /////
  //4. perform approximated convolution
  Cuda2DLaunchConfig config_conv = GetCuda2DLaunchConfig(data_entry_count, filter_weight_count, d);
  approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
      i_ind.data(), o_ind.data());

  /////
  //5. remove zero entries and convert from keys to indices
  */

  //# free memory
  cudaFree(in_ind_1d);
  cudaFree(in_ind_1d_channels);
  cudaFree(unique_masked);

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
}
}  // end namespace functor

// Instantiate the GPU implementation for float.

template struct functor::ApproxDirectSparseConvFunctor<GPUDevice, int, int>;
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