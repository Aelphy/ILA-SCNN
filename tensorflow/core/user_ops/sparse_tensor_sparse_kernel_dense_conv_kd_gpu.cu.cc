#if GOOGLE_CUDA

#define EIGEN_USE_GPU


#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename dtype>
__global__ void S2D(Cuda2DLaunchConfig config, const dtype* input_ptr,
                    dtype* output_ptr) {
  CUDA_AXIS_KERNEL_LOOP(x, config.virtual_thread_count, x) {
    if (x < 0) {  // x might overflow when testing extreme case
      break;
    }
    CUDA_AXIS_KERNEL_LOOP(y, config.virtual_thread_count, y) {
      if (y < 0) {  // y might overflow when testing extreme case
        break;
      }
      CUDA_AXIS_KERNEL_LOOP(z, config.virtual_thread_count, z) {
        if (z < 0) {  // z might overflow when testing extreme case
          break;
        }
        atomicAdd(&output_ptr[x], 1);
      }
    }
  }
}

namespace functor {
template <typename T, typename IndiceT, int DIM>
struct ApproxDirectSparseConvFunctor {
  typedef typename TTypes<T, 1>::ConstTensor ConstValT;
  typedef typename TTypes<T, 1>::Tensor ValT;
  typedef typename TTypes<T, DIM>::ConstTensor ConstIndT;
  typedef typename TTypes<T, DIM>::Tensor IndT;

  void operator()(const GPUDevice& d, ConstIndT& i_ind, ConstValT& i_val, ConstIndT& f_ind, ConstValT& f_val, IndT& o_ind, ValT& o_val) {
    const int num_data_entries = i_ind.dimension(0);
    const int num_filter_weights = f_ind.dimension(0);
    //1. set channel to 0 and convert indices to key

    //2. remove duplicates from data (e.g. thrust unique): data to search in

    //3. set up rule for more than one channel

    //4. prepare filter (to directly manipulate keys instead of indices)

    //5. perform convolution
    const int total_count = num_data_entries;
		const int filter_count = num_filter_weights;
    Cuda2DLaunchConfig config_conv = GetCuda2DLaunchConfig(total_count, filter_count, d);
    S2D<<<config_conv.block_count, config_conv.thread_per_block, 0, d.stream()>>>(config_conv,
        i_ind.data(), o_ind.data());

    //6. remove zero entries and convert from keys to indices

  }
};
}  // end namespace functor

// Instantiate the GPU implementation for float.
template struct functor::ApproxDirectSparseConvFunctor<float, int, 4>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
