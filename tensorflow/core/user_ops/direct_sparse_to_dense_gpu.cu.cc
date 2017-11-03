#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_to_dense_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

namespace tensorflow {

namespace functor {
  template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
  sparse_to_dense(CudaLaunchConfig config, const itype* __restrict__ in_id1d, const dtype* __restrict__ in_val, const itype* __restrict__ in_shape,  dtype* out_vals)
  {
    CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
      if (x < 0) {  //x might overflow when testing extreme case
        break;
      }
      out_vals[in_id1d[x]] = in_val[x];
    }
  }

  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectSparseToDenseFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
    const Tensor *in_indices, *in_values, *in_shape, *in_block_channel_mapping;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_channel_mapping));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_sh = in_shape->flat<IndiceT>();
    auto i_ind = in_indices->flat<IndiceT>();
    auto i_val = in_values->flat<T>();
    auto i_mapping = in_block_channel_mapping->flat<int>();
    auto bcount = i_mapping.dimension(0);
    int data_entry_count;
    cudaMemcpy(&data_entry_count, i_mapping.data() + bcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<IndiceT> cpu_shape(data_dimension);
    cudaMemcpy(&cpu_shape[0], i_sh.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    //LOG(INFO) << "sparse 2 dense " << data_entry_count << " : " << cpu_shape[0] << " " << cpu_shape[1] << " " << cpu_shape[2] << " " << cpu_shape[3] << " " << cpu_shape[4] << std::endl;

    /////
    //1. allocate output tensors
    int total_count = 1;
    TensorShape out_val_shape;
    for(int i = 0; i < data_dimension; ++i){
      out_val_shape.AddDim(cpu_shape[i]);
      total_count = total_count * cpu_shape[i];
    }
    Tensor *out_values;
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
    auto o_val = out_values->flat<T>();
    cudaMemset(o_val.data(), 0, total_count * sizeof(T));
    if(data_entry_count > 0){
      CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
      sparse_to_dense<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), i_val.data(), i_sh.data(), o_val.data());
    }
  }
  
  template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
  sparse_to_dense_backprops(CudaLaunchConfig config, const itype* __restrict__ in_id1d, const dtype* __restrict__ grads, const itype* __restrict__ in_shape,  dtype* backprops)
  {
    CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
      if (x < 0) {  //x might overflow when testing extreme case
        break;
      }
      backprops[x] = grads[in_id1d[x]];
    }
  }

  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectSparseToDenseBackpropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
    const Tensor *in_indices, *in_values, *in_shape, *in_block_channel_mapping, *gradients;
    OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("in_block_channel_mapping", &in_block_channel_mapping));
    OP_REQUIRES_OK(context, context->input("grads", &gradients));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_sh = in_shape->flat<IndiceT>();
    auto i_ind = in_indices->flat<IndiceT>();
    auto i_val = in_values->flat<T>();
    auto i_mapping = in_block_channel_mapping->flat<int>();
    auto bcount = i_mapping.dimension(0);
    auto grads = gradients->flat<T>();
    int data_entry_count;
    cudaMemcpy(&data_entry_count, i_mapping.data() + bcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<IndiceT> cpu_shape(data_dimension);
    cudaMemcpy(&cpu_shape[0], i_sh.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);

    //LOG(INFO) << "sparse to dense backprop";
    /////
    //1. allocate output tensors
    TensorShape out_val_shape = {i_ind.dimension(0)};
    Tensor *out_values;
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &out_values));
    auto o_val = out_values->flat<T>();
    cudaMemset(o_val.data(), 0, i_ind.dimension(0) * sizeof(T));
    if(data_entry_count > 0){
      CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
      sparse_to_dense_backprops<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), grads.data(), i_sh.data(), o_val.data());
    }
  }

  template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
  compute_block_channel_count(CudaLaunchConfig config, const dtype* __restrict__ in_val, const itype* __restrict__ in_sh, int* block_channel_count)
  {
    itype id_kd[data_dimension];
    CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
      if (x < 0) {  //x might overflow when testing extreme case
        break;
      }
      if(in_val[x] == 0) continue;
      itype idx1d = x;
      index_1DtoKD<itype, data_dimension>(0, idx1d, in_sh, &id_kd[0]);
      int bc_id = id_kd[0] * in_sh[data_dimension - 1] + id_kd[data_dimension - 1];
      atomicAdd(&block_channel_count[bc_id], 1);
    }
  }

  template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
  dense_to_sparse(CudaLaunchConfig config, const dtype* __restrict__ in_val, const itype* __restrict__ in_sh,  const int* block_channel_mapping, int* block_channel_count, itype* out_ind, dtype* out_val)
  {
    itype id_kd[data_dimension];
    CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
      if (x < 0) {  //x might overflow when testing extreme case
        break;
      }
      if(in_val[x] == 0) continue;
      itype idx1d = x;
      index_1DtoKD<itype, data_dimension>(0, idx1d, in_sh, &id_kd[0]);
      int bc_id = id_kd[0] * in_sh[data_dimension - 1] + id_kd[data_dimension - 1];
      int out_id = atomicAdd(&block_channel_count[bc_id], 1);
      int offset = block_channel_mapping[bc_id];
      out_ind[out_id + offset] = x;
      out_val[out_id + offset] = in_val[x];
    }
  }

  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectDenseToSparseFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
    const Tensor *in_values;
    OP_REQUIRES_OK(context, context->input("in_values", &in_values));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_val_ = in_values->flat_inner_dims<T, data_dimension>();
    auto i_val = in_values->flat<T>();
    std::vector<IndiceT> cpu_in_shape(data_dimension);
    int dense_entry_count = 1;
    for(int i = 0; i < data_dimension; ++i){
      cpu_in_shape[i] = (IndiceT) i_val_.dimension(i);
      dense_entry_count *= i_val_.dimension(i);
    }
    const int batch_count = cpu_in_shape[0];
    const int channel_count = cpu_in_shape[data_dimension - 1];
    Tensor *out_shape, *out_block_channel_mapping;
    TensorShape out_sh_shape = {(IndiceT) data_dimension};
    TensorShape out_bcm_shape = {(IndiceT) batch_count * channel_count + 1};
    OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
    OP_REQUIRES_OK(context, context->allocate_output("out_block_channel_mapping", out_bcm_shape, &out_block_channel_mapping));
    auto o_sh = out_shape->flat<IndiceT>();
    auto o_mapping = out_block_channel_mapping->flat<int>();
    cudaMemcpy(o_sh.data(), &cpu_in_shape[0], data_dimension * sizeof(IndiceT), cudaMemcpyHostToDevice);
    int *block_count = 0;
    Tensor block_count_tensor;
    allocate_tensor(context, block_count_tensor, &block_count, channel_count * batch_count + 1);
    cudaMemset(block_count, 0, (batch_count * channel_count + 1) * sizeof(IndiceT));
    if(dense_entry_count <= 0){
      Tensor *out_indices, *out_values;
      TensorShape out_val_shape = {(IndiceT) 0};
      TensorShape out_ind_shape = {(IndiceT) 0};
      OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
      OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
      return;
    }
    CudaLaunchConfig config = GetCudaLaunchConfig(dense_entry_count, d);
    compute_block_channel_count<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_val.data(), o_sh.data(), block_count);
    cudaStreamSynchronize(d.stream());
    compute_scan(context, d, o_mapping.data(), block_count, (batch_count * channel_count + 1), false); //exclusive scan
    int offset0, offset1;
    cudaMemcpy(&offset0, o_mapping.data() + batch_count * channel_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&offset1, block_count, sizeof(int), cudaMemcpyDeviceToHost);
    int sparse_count = offset0 + offset1;

    Tensor *out_indices, *out_values;
    TensorShape out_val_shape = {(IndiceT) sparse_count};
    TensorShape out_ind_shape = {(IndiceT) sparse_count};
    OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
    OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
    auto o_val = out_values->flat<T>();
    auto o_ind = out_indices->flat<IndiceT>();
    cudaMemset(block_count, 0, (batch_count * channel_count + 1) * sizeof(IndiceT));
    dense_to_sparse<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_val.data(),  o_sh.data(), o_mapping.data(), block_count, o_ind.data(), o_val.data());
  }
  
  
  template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
  dense_to_sparse_backprops(CudaLaunchConfig config, const itype* __restrict__ in_id1d, const dtype* __restrict__ grads, const itype* __restrict__ in_shape,  dtype* backprops)
  {
    CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
      if (x < 0) {  //x might overflow when testing extreme case
        break;
      }
      backprops[in_id1d[x]] = grads[x];
    }
  }

  template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
  void DirectDenseToSparseBackpropFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
    const Tensor *in_indices, *in_values, *in_shape, *in_block_channel_mapping, *gradients;
    OP_REQUIRES_OK(context, context->input("out_indices", &in_indices));
    OP_REQUIRES_OK(context, context->input("out_values", &in_values));
    OP_REQUIRES_OK(context, context->input("out_shape", &in_shape));
    OP_REQUIRES_OK(context, context->input("out_block_channel_mapping", &in_block_channel_mapping));
    OP_REQUIRES_OK(context, context->input("grads", &gradients));
    const DeviceT d = context->eigen_device<DeviceT>();
    auto i_sh = in_shape->flat<IndiceT>();
    auto i_ind = in_indices->flat<IndiceT>();
    auto i_val = in_values->flat<T>();
    auto i_mapping = in_block_channel_mapping->flat<int>();
    auto bcount = i_mapping.dimension(0);
    auto grads = gradients->flat<T>();
    int data_entry_count;
    cudaMemcpy(&data_entry_count, i_mapping.data() + bcount - 1, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<IndiceT> cpu_shape(data_dimension);
    cudaMemcpy(&cpu_shape[0], i_sh.data(), (data_dimension) * sizeof(IndiceT), cudaMemcpyDeviceToHost);
    int dense_entry_count = 1;
    for(int i = 0; i < data_dimension; ++i){
      dense_entry_count *= cpu_shape[i];
    }

    /////
    //1. allocate output tensors
    TensorShape out_val_shape = {dense_entry_count};
    Tensor *out_values;
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &out_values));
    auto o_val = out_values->flat<T>();
    cudaMemset(o_val.data(), 0, dense_entry_count * sizeof(T));
    if(data_entry_count > 0){
      CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
      dense_to_sparse_backprops<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), grads.data(), i_sh.data(), o_val.data());
    }
  }
} //end namespace functor


#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseToDenseFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseToDenseBackpropFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectDenseToSparseFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectDenseToSparseBackpropFunctor<GPUDevice, type, indice_type, dim>;
#define INIT_GPU_ALL(type, dim)    \
  INIT_GPU_TYPE(type, int64, dim); \
  INIT_GPU_TYPE(type, int32, dim);
#define INIT_GPU_ALL_(type)    \
  INIT_GPU_ALL(type, 5);

INIT_GPU_ALL_(float);
#undef EPS__
#undef INIT_GPU_TYPE
#undef INIT_GPU_ALL
#undef INIT_GPU_ALL_
} // end namespace tensorflow
#endif  // GOOGLE_CUDA
