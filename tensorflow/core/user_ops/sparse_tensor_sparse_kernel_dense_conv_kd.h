#pragma once

#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

  //voting sheme convolution (see vote3d)
  template <typename IndexT, typename ValueT, typename ShapeT, typename FIndexT, typename FValueT, typename FShapeT, typename T> void
  sparseCuboidConvKD(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const FIndexT& f_ind, 
                        const FValueT& f_vals, 
                        const FShapeT& f_sh,
                        const std::vector<int32>& stride_,
                        const int64 dim,
                        std::map<std::vector<int64>, T>& output_map,
                        std::vector<int64>& out_shape,
                        const std::string padding="SAME") {
    int padding_type = 1;
    if(padding == "SAME") padding_type = 0;

    const int id_in_batch = 0, id_in_depth = 1, id_in_width = id_in_depth + dim - 1, id_in_in_channels = id_in_depth + dim;
    const int id_f_depth = 0, id_f_width = id_f_depth + dim - 1, id_f_in_channels = id_f_depth + dim, id_f_out_channels = id_f_depth + dim + 1;

    //preparation: find center of filter
    std::vector<int64> filter_offset(f_sh.dimension(0), 0);
    out_shape.assign(in_sh.dimension(0), 0);

    //input: [batch, depth, height, width, in_channels] 
    //filter: [depth, height, width, output_channels, in_channels]
    for(int64 i = 0; i < filter_offset.size(); ++i){
      if(i >= id_in_depth && i <= id_in_width){
        filter_offset[i] = (f_sh(i-1) - 1) / 2; //TODO: precompute filter indices with offset?
        if(padding_type == 0){ //SAME: zero padding
          out_shape[i] = ceil(float(in_sh(i)) / float(stride_[i]));
        } else { //VALID: no padding
          out_shape[i] = ceil(float(in_sh(i) - f_sh(i-1) + 1) / float(stride_[i]));
          if(out_shape[i] < 1) out_shape[i] = 1;
        }
      } else if(i == id_in_in_channels){
        out_shape[i] = f_sh(id_f_out_channels);
      } else {
        out_shape[i] = in_sh(i);
      }
    }

    //use same pattern for stride as tensorflows dense convolution
    std::vector<int> str_padding_offset(in_ind.dimension(1), 0);
    for(int64 i = 0; i < str_padding_offset.size(); ++i){
      if(padding_type == 0){ //SAME: zero padding
        if(int(in_sh(i)) % stride_[i] == 0){
          str_padding_offset[i] = 1;
        }
      } else { //VALID: no padding
        str_padding_offset[i] = 0;
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
          int64 out_plain_id = (int64)(in_ind(i,l) - f_ind(j,k) + filter_offset[l]);
          if(padding_type == 1){ //valid padding
            out_plain_id = out_plain_id - filter_offset[l];
          }
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

  //normal convolution in sparse data
  template <typename IndexT, typename FIndexT, typename ValueT, typename ShapeT, typename KeyT, typename DataT>
  class ConvKDHelper {
  public:
    ConvKDHelper( const IndexT* data_index, 
                  const ValueT* data_value, 
                  const ShapeT* data_shape, 
                  const FIndexT* filter_index, 
                  const ValueT* filter_weights, 
                  const ShapeT* filter_shape, 
                  const std::vector<int32>* stride,
                  std::string padding_type,
                  int dimension = 3)
      : m_data_index(data_index), m_data_value(data_value), m_data_shape(data_shape), m_filter_index(filter_index), 
        m_filter_weights(filter_weights), m_filter_shape(filter_shape), m_stride(stride), m_dimension(dimension)
    {
      m_padding_type = 0; //SAME: zero padding
      if(padding_type == "VALID") m_padding_type = 1;
      for(size_t i = 0; i < m_data_index->dimension(0); ++i){
        KeyT id(m_data_index->dimension(1));
        for(size_t j = 0; j < m_data_index->dimension(1); ++j){
          id[j] = (*m_data_index)(i,j);
        }
        m_map.insert(std::make_pair(id, (*data_value)(i)));
      }

      m_filter_offset.assign(data_shape->dimension(0), 0);
      for(int i = 1; i < m_dimension + 1; ++i){
        m_filter_offset[i] = ((*filter_shape)(i-1) - 1) / 2;
      }
    }

    inline DataT
    evaluate_at(const KeyT& a_id) const {
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, output_channels, in_channels]
      DataT conv_value = 0;
      for(size_t i = 0; i < m_filter_index->dimension(0); ++i){
        if(a_id[m_dimension + 1] != (*m_filter_index)(i, m_dimension + 1)) continue; //different channels
        KeyT key_look_up = a_id;
        for(size_t j = 1; j < m_dimension + 1; ++j){
          key_look_up[j] = a_id[j] - m_filter_offset[j] + (*m_filter_index)(i,j - 1);
        }
        auto it = m_map.find(key_look_up);
        if(it != m_map.end()){
          conv_value += it->second;
        }
      }
      return conv_value;
    }

  protected:
    const IndexT* m_data_index;
    const ValueT* m_data_value;
    const ShapeT* m_data_shape;
    const FIndexT* m_filter_index;
    const ValueT* m_filter_weights;
    const ShapeT* m_filter_shape;
    const std::vector<int32>* m_stride;
    int m_padding_type;
    int m_dimension;
    std::vector<int64> m_filter_offset;
    std::map<KeyT, DataT> m_map;
  };
}

