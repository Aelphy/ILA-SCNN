#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_pooling_kd_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

//TODO: support same SAME and UNPADDED convolutions

namespace tensorflow {

//Compress [batch, x, y, ...] indices into a [1D] key while voxelization
template <typename dtype, int data_dimension> __global__ void  __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
compute_voxel_id1D__(CudaLaunchConfig config, const dtype* __restrict__ in_ptr_1d, const dtype* __restrict__ in_shape_ptr, const dtype* __restrict__ out_shape_ptr,
                    dtype* out_ind_ptr, const int* voxel_sizes_){
  dtype idx_kd[data_dimension];
  int voxel_sizes[data_dimension];
  for(int i = 0; i < data_dimension; ++i){
    voxel_sizes[i] = voxel_sizes_[i];
  }
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if(x < 0){  //x might overflow when testing extreme case
      break;
    } 
    index_1DtoKD<dtype, data_dimension>(0, in_ptr_1d[x], in_shape_ptr, &idx_kd[0]);
    for(int i = 0; i < data_dimension; ++i){
      idx_kd[i] = dtype(floor(float(idx_kd[i]) / voxel_sizes[i])) * voxel_sizes[i];
    }
    dtype val = 0;
    index_KDto1D_<dtype, data_dimension>(&idx_kd[0], out_shape_ptr, &val);
    out_ind_ptr[x] = val;
  }
}



namespace functor {
  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectSparseMaxPoolingFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context, const std::vector<int32>& stride) const {
    const Tensor *in_indices, *in_values, *in_shape, *in_block_channel_mapping;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_channel_mapping));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_sh = in_shape->flat<IndiceT>();
    auto i_ind = in_indices->matrix<IndiceT>();
    auto i_val = in_values->flat<T>();
    auto i_mapping = in_block_channel_mapping->flat<int>();
    auto bcount = i_mapping.dimension(0);
    int data_entry_count;
    cudaMemcpy(&data_entry_count, i_mapping.data() + bcount, sizeof(int), cudaMemcpyDeviceToHost);
    
    //allocate temp buffer
    Tensor in_out_map_tensor, in_out_map_ids_sorted_tensor, out_sorted_values_tensor, strides_tensor; 
    IndiceT *in_out_map_ids = 0, *in_out_map_ids_sorted = 0;
    int32* strides_ = 0;
    T *out_sorted_values = 0;
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    allocate_tensor(context, in_out_map_tensor, &in_out_map_ids, data_entry_count);
    allocate_tensor(context, in_out_map_ids_sorted_tensor, &in_out_map_ids_sorted, data_entry_count);
    allocate_tensor(context, out_sorted_values_tensor, &out_sorted_values, data_entry_count);
    allocate_tensor(context, strides_tensor, &strides_, data_dimension);
    cudaMemcpy(&strides_, &stride[0], data_dimension * sizeof(int32), cudaMemcpyHostToDevice);
    compute_voxel_id1D__<IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), i_sh.data(), i_sh.data()/*TODO*/, in_out_map_ids, strides_);
    compute_sort(context, d, in_out_map_ids, in_out_map_ids_sorted, i_val.data(), out_sorted_values, data_entry_count);
    
    Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL, *out_block_mapping = NULL;
    TensorShape out_ind_shape = {(IndiceT) 1};
    TensorShape out_val_shape = {(IndiceT) 1};
    TensorShape out_block1_shape = {(IndiceT) 1};
    TensorShape out_sh_shape = {(IndiceT) 1};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_block_channel_mapping", out_block1_shape, &out_block_mapping));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
    auto o_sh = out_shape->flat<IndiceT>();
    auto o_ind = out_indices->flat<IndiceT>();
    auto o_mapping = out_block_mapping->flat<int>();
    auto o_val = out_values->flat<T>();
  }
} // end namespace functor

#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseMaxPoolingFunctor<GPUDevice, type, indice_type, dim>;
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
