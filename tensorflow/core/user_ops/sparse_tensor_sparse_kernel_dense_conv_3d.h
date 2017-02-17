#pragma once

#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

  template<typename Matrix2DT>  
  inline bool index_is_unaffected_by_stride(const Matrix2DT& ids, const std::vector<int32>& stride, const int& a_id) {
    //assert(id.size() == stride.size());
    for(int32 i = 0; i < stride.size(); ++i){
      if(stride[i] > 1){
        if(((ids(a_id,i) + 1) % stride[i]) == 0){
          return false;
        }
      }
    }
    return true;
  }


  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename T> void
  sparseCuboidConv3D(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const FIndexT& f_ind, 
                        const FValueT& f_vals, 
                        const FShapeT& f_sh,
                        const std::vector<int32>& stride_,
                        std::map<std::vector<int64>, T>& output_map) {

    //preparation: find center of filter
    std::vector<int64> filter_offset(f_ind.dimension(1));
    for(int64 i = 0; i < filter_offset.size(); ++i){
      filter_offset[i] = (f_sh(i) - 1) / 2; //TODO: precompute filter indices with offset? 
    }

    for(int64 i = 0; i < in_ind.dimension(0); ++i){ //TODO: parallelize filtering
      if(!index_is_unaffected_by_stride(in_ind, stride_, i)) continue; // go through all indices of input and check if valid (not affected by stride)

      //a) prepare filter to update output based on current value
      std::map<std::vector<int64>, T> filter_update;
      std::vector<int64> iids(in_ind.dimension(1));
      for(int64 j = 0; j < in_ind.dimension(1); j++){
        iids[j] = in_ind(i,j);
      }
      auto &input_value = in_vals(i);
      for(int64 j = 0; j < f_ind.dimension(0); j++){
        if(f_ind(j, 1) != in_ind(i,0)) continue; //filter channel != input channel
        bool is_in_bound = true;
        std::vector<int64> update_ids(in_ind.dimension(1), 0);
        assert(update_ids.size() >= f_ind.dimension(1) - 1);
        update_ids[0] = f_ind(j,0); //output channel is filter number
        for(int64 k = 2; k < f_ind.dimension(1); k++){ //TODO: ugly coding style... prototype
          update_ids[k - 1] = iids[k - 1] + f_ind(j,k) - filter_offset[k];  //depth, width and height
          if(update_ids[k - 1] < 0 || update_ids[k - 1] >= in_sh(k-1)){ //check boundaries
            is_in_bound = false;
            break;
          }
        }
        if(is_in_bound){
          T update_val = input_value * f_vals(j); //input value times filter weight at index
          filter_update.insert(std::make_pair(update_ids, update_val));
        }
      }
      

      //b) check if output exists (search required) and create or update output based on filter_update
        //TODO: concept: tree search to find upper and lower filter bound; binary search in between?
      for(auto it = filter_update.begin(); it != filter_update.end(); ++it){
        auto res_it = output_map.find(it->first);
        if(res_it == output_map.end()){
          output_map.insert(*it);
        } else {
          res_it->second += it->second;
        }
      }
    }
  }
}

