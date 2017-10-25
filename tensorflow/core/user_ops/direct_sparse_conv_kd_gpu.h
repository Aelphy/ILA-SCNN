#pragma once

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif


#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

  namespace functor {
    template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
    struct DirectSparseConvFunctor {
      void operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, const float max_density, const std::string& filter_type) const;
    };

    template <typename DeviceT, typename T, typename IndiceT, int data_dimension>
    struct DirectSparseConvBackPropFunctor {
      void operator()(OpKernelContext* context, const std::vector<int32>& stride, const std::string& padding, const float max_density, const std::string& filter_type) const;
    };
  } //namespace functor
} //namespace tensorflow
