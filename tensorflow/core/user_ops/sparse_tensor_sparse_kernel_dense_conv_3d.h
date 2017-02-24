#pragma once

#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename T> void
  sparseCuboidConv3D(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const FIndexT& f_ind, 
                        const FValueT& f_vals, 
                        const FShapeT& f_sh,
                        const std::vector<int32>& stride_,
                        std::map<std::vector<int64>, T>& output_map,
                        std::vector<int64>& out_shape) {

    const int id_in_batch = 0, id_in_depth = 1, id_in_height = 2, id_in_width = 3, id_in_in_channels = 4;
    const int id_f_depth = 0, id_f_height = 1, id_f_width = 2, id_f_in_channels = 3, id_f_out_channels = 4;

    //preparation: find center of filter
    std::vector<int64> filter_offset(f_ind.dimension(1), 0);
    out_shape.assign(in_ind.dimension(1), 0);

    //input: [batch, depth, height, width, in_channels] 
    //filter: [depth, height, width, output_channels, in_channels]
    for(int64 i = 0; i < filter_offset.size(); ++i){
      filter_offset[i] = (f_sh(i) - 1) / 2; //TODO: precompute filter indices with offset?
      if(i >= id_in_depth && i <= id_in_width){
        out_shape[i] = ceil(float(in_sh(i)) / float(stride_[i]));
      } else if(i == id_in_in_channels){
        out_shape[i] = f_sh(id_f_out_channels);
      } else {
        out_shape[i] = in_sh(i);
      }
    }

    //use same pattern for stride as tensorflows dense convolution
    std::vector<int> str_padding_offset(in_ind.dimension(1), 0);
    for(int64 i = 0; i < str_padding_offset.size(); ++i){
      if(int(in_sh(i)) % stride_[i] == 0){
        str_padding_offset[i] = 1;
      }
    }

    //TODO: use batch in parallel? (needs more ram than a parallelization of conv)
    for(int64 i = 0; i < in_ind.dimension(0); ++i){ //TODO: parallelize filtering
      //a) prepare filter to update output based on current value
      std::map<std::vector<int64>, T> filter_update;
      for(int64 j = 0; j < f_ind.dimension(0); ++j){
        if(f_ind(j, id_f_in_channels) != in_ind(i,id_in_in_channels)) continue; //filter channel != input channel
        bool is_valid = true;
        std::vector<int64> update_ids(in_ind.dimension(1), 0);
        update_ids[id_in_batch] = in_ind(i,id_in_batch); //output channel is filter number
        update_ids[id_in_in_channels] = f_ind(j,id_f_out_channels); //output channel is filter number
        for(int64 k = id_f_depth, l = id_in_depth; k <= id_f_width; ++k, ++l){ //TODO: ugly coding style... prototype
          int64 out_plain_id = (int64)(in_ind(i,l) - f_ind(j,k) + filter_offset[k]);
          if(in_sh(l) > 1 && stride_[l] > 1){
            if(((out_plain_id ) % stride_[l]) != str_padding_offset[l]){
              is_valid = false;
              break;          
            }
            update_ids[l] = float(out_plain_id) / stride_[l]; //depth, width and height
          } else {
            update_ids[l] = out_plain_id; //depth, width and height
          }
          if(update_ids[l] < 0 || update_ids[l] >= out_shape[l]){    //check boundaries
            is_valid = false;
            break;
          }
        }
        if(is_valid){
          T update_val = in_vals(i) * f_vals(j); //input value times filter weight at index
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

