#pragma once

#include <omp.h>
#include <map>
#include <sstream>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "indexing.h"
#include "hash_map.h"

namespace tensorflow {

  //voting sheme convolution (see vote3d)
  template <typename IndexT, typename ValueT, typename ShapeT, typename T> void
  sparseCuboidConvKD(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const IndexT& f_ind, 
                        const ValueT& f_vals, 
                        const ShapeT& f_sh,
                        const std::vector<int32>& stride_,
                        const int64 dim,
                        std::vector<std::vector<int64> >& output_keys,
                        std::vector<T>& output_values,
                        std::vector<int64>& out_shape,
                        const std::string padding="SAME") {
    int padding_type = 1;
    if(padding == "SAME") padding_type = 0;

    auto in_ind_ptr = &in_ind; auto in_vals_ptr = &in_vals; auto f_ind_ptr = &f_ind; auto f_vals_ptr = &f_vals; auto in_sh_ptr = &in_sh;

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

    LFMap<std::vector<int64>, T> map(out_shape);
    auto map_ptr = &map;

#pragma omp parallel for firstprivate(in_ind_ptr, f_ind_ptr, in_vals_ptr, f_vals_ptr, in_sh_ptr, map_ptr)
    for(int64 i = 0; i < (*in_ind_ptr).dimension(0); ++i){ //TODO: parallelize filtering
      //a) prepare filter to update output based on current value
      std::vector<std::pair<int64,T> > filter_update;
      std::vector<int64> update_ids((*in_ind_ptr).dimension(1), 0);
      for(int64 j = 0; j < (*f_ind_ptr).dimension(0); ++j){
        if((*f_ind_ptr)(j, id_f_in_channels) != (*in_ind_ptr)(i,id_in_in_channels)) continue; //filter channel != input channel
        bool is_valid = true;
        update_ids[id_in_batch] = (*in_ind_ptr)(i,id_in_batch); //output channel is filter number
        update_ids[id_in_in_channels] = (*f_ind_ptr)(j,id_f_out_channels); //output channel is filter number
        for(int64 k = id_f_depth, l = id_in_depth; k <= id_f_width; ++k, ++l){ //TODO: ugly coding style... prototype
          int64 out_plain_id = (int64)((*in_ind_ptr)(i,l) - (*f_ind_ptr)(j,k) + filter_offset[l]);
          if(padding_type == 1){ //valid padding
            out_plain_id = out_plain_id - filter_offset[l];
          }
          if((*in_sh_ptr)(l) > 1 && stride_[l] > 1){
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
          T update_val = (*in_vals_ptr)(i) * (*f_vals_ptr)(j); //input value times filter weight at index
          map_ptr->update(update_ids, update_val);
        }
      }
    }
    map.traverse(output_keys,output_values);
  }

  //normal convolution in sparse data
  template <typename IndexT, typename FIndexT, typename ValueT, typename KeyT, typename DataT>
  class ConvKDHelper {
  public:
    ConvKDHelper( const IndexT* data_index, 
                  const ValueT* data_value, 
                  const FIndexT* filter_index, 
                  const ValueT* filter_weights, 
                  const std::vector<int32>* stride,
                  std::string padding_type,
                  std::vector<int64> filter_offset,
                  int dimension = 3)
      : m_data_index(data_index), m_data_value(data_value), m_filter_index(filter_index), 
        m_filter_weights(filter_weights),m_stride(stride), m_filter_offset(filter_offset), m_dimension(dimension)
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
    }

    inline DataT
    backprop_filter_at(const KeyT& a_id) const {
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, in_channels, out_channels]
      DataT conv_value = 0;
      for(size_t i = 0; i < m_filter_index->dimension(0); ++i){
        if((*m_filter_index)(i,m_dimension + 1) != a_id[m_dimension + 1]) continue; //channel of output == output channel of filter
        KeyT key_look_up(a_id.size(), 0);
        key_look_up[0] = (*m_filter_index)(i,0);
        for(size_t j = 1; j < m_dimension + 1; ++j){
          key_look_up[j] = a_id[j - 1] - m_filter_offset[j] + (*m_filter_index)(i,j);
        }
        key_look_up[m_dimension + 1] = a_id[m_dimension]; //channel of input == input channel of filter
        auto it = m_map.find(key_look_up);

        if(it != m_map.end()){
          auto data_val = it->second;
          auto weights = (*m_filter_weights)(i);
          conv_value += data_val * weights;
        }
      }
      return conv_value;
    }

    inline DataT
    backprop_indices_at(const KeyT& a_id) const {
      //input: [batch, depth, height, width, in_channels] 
      //filter: [depth, height, width, in_channels, out_channels]
      DataT conv_value = 0;
      for(size_t i = 0; i < m_filter_index->dimension(0); ++i){
        if((*m_filter_index)(i,m_dimension) != a_id[m_dimension + 1]) continue; //channel of input == input channel of filter
        KeyT key_look_up(a_id.size(), 0);
        key_look_up[0] = a_id[m_dimension]; //batch
        for(size_t j = 1; j < m_dimension + 1; ++j){
          key_look_up[j] = a_id[j - 1] - m_filter_offset[j] + (*m_filter_index)(i,j-1);
        }
        key_look_up[m_dimension + 1] = (*m_filter_index)(i,m_dimension + 1); //channel of output == output channel of filter
        auto it = m_map.find(key_look_up);

        if(it != m_map.end()){
          auto data_val = it->second;
          auto weights = (*m_filter_weights)(i);
          conv_value += data_val * weights;
        }
      }
      return conv_value;
    }

  protected:
    const IndexT* m_data_index;
    const ValueT* m_data_value;
    const FIndexT* m_filter_index;
    const ValueT* m_filter_weights;
    const std::vector<int32>* m_stride;
    int m_padding_type;
    int m_dimension;
    std::vector<int64> m_filter_offset;
    std::map<KeyT, DataT> m_map;
  };


  template<typename T> void 
  atomic_add_cas(std::atomic<T>& current_val, T add_val) {
    auto current = current_val.load();
    while (!current_val.compare_exchange_weak(current, current + add_val));
  }

  //voting sheme convolution (see vote3d)
  template <typename IndexT, typename ValueT, typename ShapeT, typename T, typename OutT> void
  sparseCuboidConvKDBackpropV2(   const IndexT& in_ind, 
                        const ValueT& in_vals, 
                        const ShapeT& in_sh, 
                        const IndexT& f_ind, 
                        const ValueT& f_vals, 
                        const ShapeT& f_sh,
                        const IndexT& grad_ind,
                        const ValueT& grad_val,
                        const ShapeT& grad_sh,
                        const std::vector<int32>& stride_,
                        const int64 dim,
                        OutT& filter_backprop,
                        OutT& ind_backprop,
                        const std::string padding="SAME") {
    int padding_type = 1;
    if(padding == "SAME") padding_type = 0;

    auto in_ind_ptr = &in_ind; auto in_vals_ptr = &in_vals; auto f_ind_ptr = &f_ind; auto f_vals_ptr = &f_vals; auto in_sh_ptr = &in_sh;

    const int id_in_batch = 0, id_in_depth = 1, id_in_width = id_in_depth + dim - 1, id_in_in_channels = id_in_depth + dim;
    const int id_f_depth = 0, id_f_width = id_f_depth + dim - 1, id_f_in_channels = id_f_depth + dim, id_f_out_channels = id_f_depth + dim + 1;

    //preparation: find center of filter
    std::vector<int64> filter_offset(f_sh.dimension(0), 0);
    std::vector<int64> out_shape(in_sh.dimension(0), 0);

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

    //build map to look up gradients
    LFMap<std::vector<int64>, T> map(out_shape);
    auto map_ptr = &map; auto grad_ind_ptr = &grad_ind; auto grad_val_ptr = &grad_val;
#pragma omp parallel for firstprivate(map_ptr, grad_ind_ptr, grad_val_ptr)
    for(int64 i = 0; i < (*grad_ind_ptr).dimension(0); ++i){
      std::vector<int64> update_ids((*grad_ind_ptr).dimension(1), 0);
      for(size_t j = 0; j < (*grad_ind_ptr).dimension(1); ++j){
        update_ids[j] = (*grad_ind_ptr)(i,j);
      }
      map_ptr->insert(update_ids, (*grad_val_ptr)(i));
    }

    std::vector<std::atomic<T> > filter_bp((*f_ind_ptr).dimension(0));
    std::vector<std::atomic<T> > input_bp((*in_ind_ptr).dimension(0));
    auto filter_bp_ptr = &filter_bp; auto input_bp_ptr = &input_bp;

//TODO: extremely ugly
//how to initialize with atomic(0) values?
#pragma omp parallel for firstprivate(input_bp_ptr)
    for(int64 i = 0; i < input_bp_ptr->size(); ++i){
      (*input_bp_ptr)[i] = 0;
    }
#pragma omp parallel for firstprivate(filter_bp_ptr)
    for(int64 i = 0; i < filter_bp_ptr->size(); ++i){
      (*filter_bp_ptr)[i] = 0;
    }
    
     //look up gradients and compute backprop
#pragma omp parallel for firstprivate(in_ind_ptr, f_ind_ptr, in_vals_ptr, f_vals_ptr, in_sh_ptr, map_ptr, input_bp_ptr, filter_bp_ptr)
    for(int64 i = 0; i < (*in_ind_ptr).dimension(0); ++i){
      //a) prepare filter to update output based on current value
      std::vector<std::pair<int64,T> > filter_update;
      std::vector<int64> update_ids((*in_ind_ptr).dimension(1), 0);
      for(int64 j = 0; j < (*f_ind_ptr).dimension(0); ++j){
        if((*f_ind_ptr)(j, id_f_in_channels) != (*in_ind_ptr)(i,id_in_in_channels)) continue; //filter channel != input channel
        bool is_valid = true;
        update_ids[id_in_batch] = (*in_ind_ptr)(i,id_in_batch); //output channel is filter number
        update_ids[id_in_in_channels] = (*f_ind_ptr)(j,id_f_out_channels); //output channel is filter number
        for(int64 k = id_f_depth, l = id_in_depth; k <= id_f_width; ++k, ++l){ //TODO: ugly coding style... prototype
          int64 out_plain_id = (int64)((*in_ind_ptr)(i,l) - (*f_ind_ptr)(j,k) + filter_offset[l]);
          if(padding_type == 1){ //valid padding
            out_plain_id = out_plain_id - filter_offset[l];
          }
          if((*in_sh_ptr)(l) > 1 && stride_[l] > 1){
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
          T grad;
          bool res = map_ptr->find(update_ids, grad);
          //assert(res); // programming mistake if there is no gradient for op?!
          T input_val_up = (*in_vals_ptr)(i) * grad;
          T filter_val_up = (*f_vals_ptr)(j) * grad; //input value times filter weight at index
          atomic_add_cas((*input_bp_ptr)[i], filter_val_up);
          atomic_add_cas((*filter_bp_ptr)[j], input_val_up);
        }
      }
    }
    auto filter_backprop_ptr = &filter_backprop;
    auto ind_backprop_ptr = &ind_backprop;
#pragma omp parallel for firstprivate(input_bp_ptr, ind_backprop_ptr)
    for(int64 i = 0; i < input_bp_ptr->size(); ++i){
      (*ind_backprop_ptr)(i) = (*input_bp_ptr)[i];
    }
#pragma omp parallel for firstprivate(filter_bp_ptr, filter_backprop_ptr)
    for(int64 i = 0; i < filter_bp_ptr->size(); ++i){
      (*filter_backprop_ptr)(i) = (*filter_bp_ptr)[i];
    }
  }
    
}

