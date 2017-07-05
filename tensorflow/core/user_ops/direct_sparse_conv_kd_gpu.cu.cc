#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "direct_sparse_conv_kd_gpu.h"

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

template <typename dtype>
__global__ void index_1DtoKD(CudaLaunchConfig config, const dtype* in_ptr, const dtype* in_shape_ptr /*[batch, dim1, ..., dimx, channel_nr]*/,
                    dtype* out_ind_ptr, dtype* out_channels_ptr, const int dimension_count /*excluding batch and channel_nr*/){
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    dtype val = 0;
    dtype channel = 0;
    for(dtype i = 0; i < dimension_count + 1; ++i) {
      //TODO 
    }
    out_ind_ptr[x] = val;
    out_channels_ptr[x] = channel;
  }
}

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

//taken from: Mark Harris, "Parallel Prefix Sum (Scan) with CUDA" (mharris@nvidia.com) 
template <typename dtype, typename itype>
__global__ void prescan(dtype *g_odata, dtype *g_idata, itype n) 
{ 
  extern __shared__ dtype temp[];
  // allocated on invocation 
  itype thid = threadIdx.x; 
  itype offset = 1; 
  temp[2*thid]   = g_idata[2*thid]; 
  // load input into shared memory 
  temp[2*thid+1] = g_idata[2*thid+1]; 
  for(itype d = n>>1; d > 0; d >>= 1) // build sum in place up the tree 
  { 
    __syncthreads(); 
    if(thid < d) { 
      itype ai = offset*(2*thid+1)-1; 
      itype bi = offset*(2*thid+2)-1; 
      temp[bi] += temp[ai];         
    } 
    offset *= 2; 
  } 
  if(thid == 0) { 
    temp[n - 1] = 0; 
  } 
  // clear the last element 
  for(itype d = 1; d < n; d *= 2) // traverse down tree & build scan 
  { 
    offset >>= 1; 
    __syncthreads(); 
    if(thid < d) 
    { 
      itype ai = offset*(2*thid+1)-1; 
      itype bi = offset*(2*thid+2)-1; 
      dtype t   = temp[ai]; 
      temp[ai]  = temp[bi]; 
      temp[bi] += t; 
    } 
  } 
  __syncthreads(); 
  g_odata[2*thid]   = temp[2*thid]; // write results to device memory 
  g_odata[2*thid+1] = temp[2*thid+1];  
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
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  const DeviceT d = context->eigen_device<DeviceT>();

  /*
  const int data_entry_count = i_ind.dimension(0);
  const int filter_weight_count = f_ind.dimension(0);
  const int channel_count = 1; //TODO

  //indices must be sorted!
  
  /////
  //1. set channel to 0 and convert indices to 1D key
  // + define rule to work with more than one channel
  IndiceT *in_ind_1d = 0;
  checkCuda(cudaMalloc(&in_ind_1d, data_entry_count * sizeof(IndiceT)));
  IndiceT *in_ind_1d_channels = 0;
  checkCuda(cudaMalloc(&in_ind_1d_channels, data_entry_count * sizeof(IndiceT)));

  CudaLaunchConfig config_i1d = GetCudaLaunchConfig(data_entry_count, d);
  index_1DtoKD<<<config_i1d.block_count, config_i1d.thread_per_block, 0, d.stream()>>>(config_i1d,
      i_ind.data(), i_sh.data(),  in_ind_1d, in_ind_1d_channels, i_ind.dimension(1));

  /////
  //2. remove duplicates from data (e.g. thrust unique) and apply stride/padding to obtain search structure
  IndiceT *search_struct = 0;
  checkCuda(cudaMalloc(&search_struct, data_entry_count * sizeof(IndiceT)));
  IndiceT search_size = 0;

  /////
  //3. prepare filter (to directly manipulate 1D keys instead of kD indices)
  
  
  /////
  //4. perform approximated convolution
  Cuda2DLaunchConfig config_conv = GetCuda2DLaunchConfig(data_entry_count, filter_weight_count, d);
  approxSparseDirectConv<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
      i_ind.data(), o_ind.data());

  /////
  //5. remove zero entries and convert from keys to indices

  //# free memory
  cudaFree(in_ind_1d);
  cudaFree(in_ind_1d_channels);
  */

  /*
  // Create an output tensor
  Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
  TensorShape out_ind_shape = {(Tindices) in_ind.dimension(0), (Tindices) in_ind.dimension(1)};
  TensorShape out_val_shape = {(Tindices) in_vals.dimension(0)};
  TensorShape out_sh_shape = {(Tindices) in_sh.dimension(0)};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &sparse_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &sparse_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &sparse_shape));
  */
  
  // Create an output tensor
  Tensor *sparse_values = NULL, *sparse_indices = NULL, *sparse_shape = NULL;
  TensorShape out_ind_shape = {1, 1};
  TensorShape out_val_shape = {1};
  TensorShape out_sh_shape = {1};
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &sparse_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &sparse_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &sparse_shape));
  

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
