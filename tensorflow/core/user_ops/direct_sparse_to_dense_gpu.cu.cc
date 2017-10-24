#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_to_dense_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

namespace tensorflow {

template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
sparse_to_dense(CudaLaunchConfig config, const itype* __restrict__ in_id1d, const dtype* __restrict__ in_val, const itype* __restrict__ in_shape,  dtype* out_vals)
{
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    out_vals[in_id1d[x]] = in_val[x];
  }
}

template<typename dtype, typename itype, int data_dimension> __global__ void __launch_bounds__(MAX_1024_THREADS_PER_BLOCK)
sparse_to_dense_backprops(CudaLaunchConfig config, const itype* __restrict__ in_id1d, const dtype* __restrict__ grads, const itype* __restrict__ in_shape,  dtype* backprops)
{
  itype id_kd[data_dimension];
  CUDA_1D_KERNEL_LOOP(x, config.virtual_thread_count) {
    if (x < 0) {  //x might overflow when testing extreme case
      break;
    }
    backprops[x] = grads[in_id1d[x]];
  }
}

namespace functor {
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

    /////
    //4. allocate output tensors
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
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    sparse_to_dense<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), i_val.data(), i_sh.data(), o_val.data());
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

    /////
    //4. allocate output tensors
    TensorShape out_val_shape = {i_ind.dimension(0)};
    Tensor *out_values;
    OP_REQUIRES_OK(context, context->allocate_output("backprops", out_val_shape, &out_values));
    auto o_val = out_values->flat<T>();
    cudaMemset(o_val.data(), 0, i_ind.dimension(0) * sizeof(T));
    CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
    sparse_to_dense_backprops<T, IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, i_ind.data(), grads.data(), i_sh.data(), o_val.data());
  }
} //end namespace functor


#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseToDenseFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseToDenseBackpropFunctor<GPUDevice, type, indice_type, dim>;
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
