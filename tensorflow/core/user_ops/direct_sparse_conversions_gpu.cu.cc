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
  std::stringstream debugs;
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
  filter_segment_count_ = filter_segment_count;
  cudaDeviceSynchronize();
  LOG(INFO) << debugs.str() << std::endl;
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
  const int data_entry_count = i_ind.dimension(0);
  int hypercube_size = 10;
  std::stringstream dout_s;
  //indices must! be sorted
  clock_t t;

  //preprocessing step (1) has to be performed only for one layer in the neural network! Also step (2) can be precomputed and shouldn't affect runtime of nn
  
  /////
  //1. Convert Coordinate List Format to sparse block format and compress k dimensional indices to 1d
  t = clock();
  int block_count = 0;
  Tensor ibi_tensor, ibp_tensor, ibpi_tensor, ibv_tensor;
  IndiceT *in_block_ids = 0, *in_block_pointer = 0, *in_block_pointer_ids = 0;
  T *in_block_vals = 0;
  coo_to_blocks<DeviceT, T, IndiceT, data_dimension>(context, d, i_ind.data(), i_val.data(), i_sh.data(), &in_block_ids, &in_block_vals, &in_block_pointer, &in_block_pointer_ids, data_entry_count, hypercube_size, block_count, ibi_tensor, ibp_tensor, ibpi_tensor, ibv_tensor);
  dout_s << "t1: " << float(clock() - t) / CLOCKS_PER_SEC << std::endl;
  LOG(INFO) << dout_s.str(); dout_s.str("");
}
}  // end namespace functor

#define INIT_GPU_TYPE(type, indice_type, dim) \
 template struct functor::DirectSparseDataConversionFunctor<GPUDevice, type, indice_type, dim>;
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
