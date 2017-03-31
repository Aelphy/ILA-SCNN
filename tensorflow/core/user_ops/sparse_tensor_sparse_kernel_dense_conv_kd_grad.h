#pragma once

#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename GradT, typename T> void
  sparseCuboidConvKDFilterGrad(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const FIndexT& f_ind, 
                        const FValueT& f_vals, 
                        const FShapeT& f_sh,
                        const GradT& grads,
                        const std::vector<int32>& stride_,
                        const int64 dim,
                        std::map<std::vector<int64>, T>& output_map) 
{
  
  
  
  
}

