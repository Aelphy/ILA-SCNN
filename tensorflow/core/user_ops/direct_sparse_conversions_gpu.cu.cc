#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <time.h>
#include <sstream>
#include "direct_sparse_conversions_gpu.h"
#include "direct_sparse_cuda_helpers_gpu.h"

namespace tensorflow {

template<typename DeviceT, typename T, typename IndiceT, int data_dimension> inline void
coo_to_blocks(  OpKernelContext* context, DeviceT d, const IndiceT* in_ids_kd, const T* in_vals, const IndiceT* in_shape, IndiceT** block_ids_1d, T** block_vals,
                  IndiceT** block_pointer, IndiceT** block_pointer_ids, int data_entry_count, int hypercube_size, int& block_count,
                  Tensor& ibi_tensor, Tensor& ibp_tensor, Tensor& ibpi_tensor, Tensor& ibv_tensor)
{
  Tensor td1_tensor, td2_tensor, td3_tensor, svi_tensor, sv_tensor;
  IndiceT *tmp_data = 0, *tmp_data2 = 0, *tmp_data3;
  IndiceT *sorted_voxel_ids = 0, *sorted_id = 0;
  allocate_tensor(context, td1_tensor, &tmp_data,  data_entry_count);
  allocate_tensor(context, td2_tensor, &tmp_data2,  data_entry_count);
  allocate_tensor(context, td3_tensor, &tmp_data3,  data_entry_count);
  allocate_tensor(context, svi_tensor, &sorted_voxel_ids,  data_entry_count);
  allocate_tensor(context, sv_tensor, &sorted_id,  data_entry_count);
  CudaLaunchConfig config = GetCudaLaunchConfig(data_entry_count, d);
  auto &voxel_id = tmp_data;
  auto &data_id = tmp_data2;
  auto &sorted_id_tmp = tmp_data3;
  compute_voxel_id1D<IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, in_shape, voxel_id, data_id, data_entry_count, hypercube_size, false);
  cudaDeviceSynchronize();
  //1. put entries per block into consecutive segments of a list
  compute_sort(context, d, voxel_id, sorted_voxel_ids /*sorted voxel ids*/, data_id, sorted_id_tmp /*sorted data ids*/, data_entry_count);
  cudaDeviceSynchronize();
  //2. set pointers to the start of each block
  auto &unique_mask = tmp_data;
  auto &unique_count = tmp_data2;
  compute_unique_mask<IndiceT><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, sorted_voxel_ids, unique_mask);
  cudaDeviceSynchronize();
  compute_scan(context, d, unique_count, unique_mask, data_entry_count);
  cudaDeviceSynchronize();
  IndiceT block_count_ = 0;
  cudaMemcpy(&block_count_, unique_count + data_entry_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  block_count = block_count_;
  checkCuda(cudaMalloc(block_pointer, (block_count_ + 1) * sizeof(IndiceT)));
  allocate_tensor(context, ibp_tensor, block_pointer,  block_count_ + 1);
  IndiceT dec = data_entry_count;
  cudaMemcpy(&(*block_pointer)[block_count_], &dec, sizeof(IndiceT), cudaMemcpyHostToDevice);
  allocate_tensor(context, ibpi_tensor, block_pointer_ids,  block_count_ + 1);
  compute_block_start<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, unique_mask, unique_count, sorted_voxel_ids, *block_pointer, *block_pointer_ids);
  cudaDeviceSynchronize();
  //3. apply block structure to data
  auto &id1d = data_id;
  index_KDto1D<IndiceT, data_dimension><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, in_ids_kd, in_shape, id1d, data_entry_count);
  allocate_tensor(context, ibi_tensor, block_ids_1d,  data_entry_count);
  allocate_tensor(context, ibv_tensor, block_vals,  data_entry_count);
  apply_sorted_indices<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, *block_ids_1d, id1d, sorted_id_tmp);
  apply_sorted_indices<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(config, *block_vals, in_vals, sorted_id_tmp);
  //# free temporary resources 
  cudaDeviceSynchronize();
}

template<typename DeviceT, typename T, typename IndiceT, int data_dimension> inline void
preprocess_filter(OpKernelContext* context, DeviceT d, const IndiceT* f_ids_kd, const T* f_vals, const IndiceT* f_shape, const IndiceT* i_shape, int filter_weight_count,
    int** filter_segments_start, int** filter_segments_end, T** filter_sorted_weights, IndiceT** filter_sorted_ind_1d, int& filter_segment_count_, int* filter_channel_mapping, int in_channel_count,
    int out_channel_count, Tensor& fss_tensor, Tensor& fse_tensor, Tensor& fsw_tensor, Tensor& fsi_tensor)
{
  Tensor um_tensor, uc_tensor, fi1d_tensor, foc_tensor, fic_tensor, fid_tensor;
  IndiceT *unique_masked = 0;
  allocate_tensor(context, um_tensor, &unique_masked,  filter_weight_count);
  IndiceT *unique_count = 0;
  allocate_tensor(context, uc_tensor, &unique_count,  filter_weight_count);
  IndiceT *filter_ind_1d = 0;
  allocate_tensor(context, fi1d_tensor, &filter_ind_1d,  filter_weight_count);
  IndiceT *filter_out_channel = 0;
  allocate_tensor(context, foc_tensor, &filter_out_channel,  filter_weight_count);
  IndiceT *filter_in_channel = 0;
  allocate_tensor(context, fic_tensor, &filter_in_channel,  filter_weight_count);
  IndiceT *filter_id = 0;
  allocate_tensor(context, fid_tensor, &filter_id,  filter_weight_count);
  CudaLaunchConfig config_f1d = GetCudaLaunchConfig(filter_weight_count, d);
  prepare_filter_weights_<IndiceT, data_dimension><<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d,
    f_ids_kd, f_shape, i_shape, filter_ind_1d, filter_out_channel, filter_in_channel, filter_id, filter_weight_count);
  cudaDeviceSynchronize();
  //sort filter w.r.t. output and input channels
  IndiceT* new_filter_indice = 0;
  IndiceT* filter_sorted_out = 0;
  IndiceT* filter_sorted_in = 0;
  IndiceT* filter_sorted_tmp_c_in = 0;
  IndiceT filter_segment_count = 0;
  Tensor nfi_tensor, tfso_tensor, tfsi_tensor, tfsc_tensor;
  allocate_tensor(context, nfi_tensor, &new_filter_indice,  filter_weight_count);
  allocate_tensor(context, tfso_tensor, &filter_sorted_out,  filter_weight_count);
  allocate_tensor(context, tfsi_tensor, &filter_sorted_in,  filter_weight_count);
  allocate_tensor(context, tfsc_tensor, &filter_sorted_tmp_c_in,  filter_weight_count);
  compute_sort(context, d, filter_out_channel, filter_sorted_out, filter_id, new_filter_indice, filter_weight_count);
  compute_unique_mask<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_out, unique_masked);
  compute_scan(context, d, unique_count, unique_masked, filter_weight_count);
  cudaMemcpy(&filter_segment_count, unique_count + filter_weight_count - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  allocate_tensor(context, fss_tensor, filter_segments_start,  filter_weight_count);
  allocate_tensor(context, fse_tensor, filter_segments_end,  filter_weight_count);
  cudaDeviceSynchronize();
  compute_segment_start<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_segments_start, unique_masked, unique_count);
  cudaDeviceSynchronize();
  compute_segment_end<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_segments_end, *filter_segments_start, unique_count, filter_weight_count);
  cudaDeviceSynchronize();
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, filter_sorted_tmp_c_in, filter_in_channel, new_filter_indice);
  cudaDeviceSynchronize();
  CudaLaunchConfig config_fi = GetCudaLaunchConfig(std::max(filter_weight_count, in_channel_count * out_channel_count + 1), d);
  //TODO: check if filter_sorted_tmp_c_in and  filter_in_channel are correct
  compute_segmented_sort(context, d, filter_sorted_tmp_c_in, filter_sorted_in, new_filter_indice, filter_id, filter_weight_count, filter_segment_count, *filter_segments_start, *filter_segments_end);
  cudaDeviceSynchronize();
  compute_filter_channel_index<<<config_fi.block_count, config_fi.thread_per_block, 0, d.stream()>>>(config_fi, filter_sorted_in, filter_sorted_out,
    filter_channel_mapping, in_channel_count, out_channel_count, filter_weight_count);
  allocate_tensor(context, fsi_tensor, filter_sorted_ind_1d,  filter_weight_count);
  allocate_tensor(context, fsw_tensor, filter_sorted_weights,  filter_weight_count);
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_sorted_weights, f_vals, filter_id);
  apply_sorted_indices<<<config_f1d.block_count, config_f1d.thread_per_block, 0, d.stream()>>>(config_f1d, *filter_sorted_ind_1d, filter_ind_1d, filter_id);
  //cudaMemcpy(*filter_sorted_ind_1d, filter_id, filter_weight_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  filter_segment_count_ = filter_segment_count;
  cudaDeviceSynchronize();
}

namespace functor {
template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseDataConversionFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  clock_t t_total = clock();
  const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
  OP_REQUIRES_OK(context, context->input("in_indices", &in_indices));
  OP_REQUIRES_OK(context, context->input("in_values", &in_values));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto i_ind = in_indices->matrix<IndiceT>();
  auto i_val = in_values->flat<T>();
  const IndiceT data_entry_count = i_ind.dimension(0);
  int hypercube_size = 16; //TODO: get as parameter
  std::stringstream dout_s;
  //indices must! be sorted
  //preprocessing step (1) has to be performed only for one layer in the neural network! Also step (2) can be precomputed and shouldn't affect runtime of nn
  
  /////
  //1. Convert Coordinate List Format to sparse block format and compress k dimensional indices to 1d
  int block_count = 0;
  Tensor ibi_tensor, ibp_tensor, ibpi_tensor, ibv_tensor;
  IndiceT *in_block_ids = 0, *in_block_pointer = 0, *in_block_pointer_ids = 0;
  T *in_block_vals = 0;
  coo_to_blocks<DeviceT, T, IndiceT, data_dimension>(context, d, i_ind.data(), i_val.data(), i_sh.data(), &in_block_ids, &in_block_vals, &in_block_pointer, &in_block_pointer_ids, data_entry_count, hypercube_size, block_count, ibi_tensor, ibp_tensor, ibpi_tensor, ibv_tensor);
  
  Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL, *data_count = NULL, *out_block_pointer = NULL, *out_block_pointer_ids = NULL;
  TensorShape out_ind_shape = {data_entry_count}; 
  TensorShape out_val_shape = {data_entry_count};
  TensorShape out_block1_shape = {(IndiceT) block_count + 1};
  TensorShape out_block1_ids_shape = {(IndiceT) block_count + 1};
  TensorShape out_sh_shape = {(IndiceT) data_dimension};
  TensorShape out_count_shape = {(IndiceT) 1}; 
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_block_ptr", out_block1_shape, &out_block_pointer));
  OP_REQUIRES_OK(context, context->allocate_output("out_block_ptr_ids", out_block1_ids_shape, &out_block_pointer_ids));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  OP_REQUIRES_OK(context, context->allocate_output("data_count", out_count_shape, &data_count));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_b_ptr = out_block_pointer->flat<IndiceT>();
  auto o_b_ptr_ids = out_block_pointer_ids->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto data_offset = data_count->flat<int>();
  cudaMemcpy(o_sh.data(), i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_ind.data(), in_block_ids, data_entry_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_b_ptr.data(), in_block_pointer, (block_count + 1) * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_b_ptr_ids.data(), in_block_pointer_ids, (block_count + 1) * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_val.data(), in_block_vals, data_entry_count * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(data_offset.data(), &data_entry_count, sizeof(IndiceT), cudaMemcpyHostToDevice);
}

template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
void DirectSparseFilterConversionFunctor<DeviceT, T, IndiceT, data_dimension>::operator()(OpKernelContext* context) const {
  clock_t t_total = clock();
  const Tensor *in_indices, *in_values, *in_shape, *filter_indices, *filter_values, *filter_shape;
  OP_REQUIRES_OK(context, context->input("filter_indices", &filter_indices));
  OP_REQUIRES_OK(context, context->input("filter_values", &filter_values));
  OP_REQUIRES_OK(context, context->input("filter_shape", &filter_shape));
  OP_REQUIRES_OK(context, context->input("in_shape", &in_shape));
  const DeviceT d = context->eigen_device<DeviceT>();
  auto i_sh = in_shape->flat<IndiceT>();
  auto f_ind = filter_indices->matrix<IndiceT>();
  auto f_val = filter_values->flat<T>();
  auto f_sh = filter_shape->flat<IndiceT>();
  const IndiceT filter_weight_count = f_ind.dimension(0);

  IndiceT out_channel_count = -1;
  cudaMemcpy(&out_channel_count, f_sh.data() + data_dimension - 1, sizeof(IndiceT), cudaMemcpyDeviceToHost);
  std::vector<IndiceT> cpu_in_shape(data_dimension);
  cudaMemcpy(&cpu_in_shape[0], i_sh.data(), data_dimension * sizeof(IndiceT), cudaMemcpyDeviceToHost);
  const int in_channel_count = cpu_in_shape[data_dimension - 1];

  Tensor fcm_tensor, fss_tensor, fse_tensor, fsw_tensor, fsi_tensor;
  int* filter_channel_mapping = 0;
  int *filter_segments_start = 0, *filter_segments_end = 0;
  int filter_segment_count_ = 0;
  allocate_tensor(context, fcm_tensor, &filter_channel_mapping,  (out_channel_count * in_channel_count + 1));
  T* filter_sorted_weights = 0;
  IndiceT* filter_sorted_ind_1d = 0;
  preprocess_filter<DeviceT, T, IndiceT, data_dimension>(context, d, f_ind.data(), f_val.data(), f_sh.data(), i_sh.data(), filter_weight_count, &filter_segments_start, &filter_segments_end, &filter_sorted_weights, &filter_sorted_ind_1d, filter_segment_count_, filter_channel_mapping, in_channel_count, out_channel_count, fss_tensor, fse_tensor, fsw_tensor, fsi_tensor);

  Tensor *out_values = NULL, *out_indices = NULL, *out_shape = NULL, *data_channel_mapping = NULL;
  TensorShape out_ind_shape = {filter_weight_count}; 
  TensorShape out_val_shape = {filter_weight_count};
  TensorShape out_sh_shape = {(IndiceT) data_dimension};
  TensorShape out_mapping_shape = {(int) (out_channel_count * in_channel_count + 1)}; 
  OP_REQUIRES_OK(context, context->allocate_output("out_indices", out_ind_shape, &out_indices));
  OP_REQUIRES_OK(context, context->allocate_output("out_values", out_val_shape, &out_values));
  OP_REQUIRES_OK(context, context->allocate_output("out_shape", out_sh_shape, &out_shape));
  OP_REQUIRES_OK(context, context->allocate_output("out_channel_mapping", out_mapping_shape, &data_channel_mapping));
  auto o_sh = out_shape->flat<IndiceT>();
  auto o_ind = out_indices->flat<IndiceT>();
  auto o_val = out_values->flat<T>();
  auto o_cmap = data_channel_mapping->flat<int>();
  cudaMemcpy(o_sh.data(), f_sh.data(), sizeof(IndiceT) * data_dimension, cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_ind.data(), filter_sorted_ind_1d, filter_weight_count * sizeof(IndiceT), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_val.data(), filter_sorted_weights, filter_weight_count * sizeof(T), cudaMemcpyDeviceToDevice);
  cudaMemcpy(o_cmap.data(), filter_channel_mapping, (out_channel_count * in_channel_count + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
}
}  // end namespace functor

#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseDataConversionFunctor<GPUDevice, type, indice_type, dim>; \
 template struct functor::DirectSparseFilterConversionFunctor<GPUDevice, type, indice_type, dim>;
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
