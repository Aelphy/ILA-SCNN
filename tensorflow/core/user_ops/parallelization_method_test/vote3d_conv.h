#pragma once

#include <vector>
#include <string>
#include <math.h>

template <typename KeyTypeT, typename ValueT, typename WeightT>
class SparseConvUpdate {
public:
    SparseConvUpdate( const std::vector<KeyTypeT>& a_kernel_ids,
                      const std::vector<WeightT>& a_kernel_weights,
                      const KeyTypeT& a_kernel_shape,
                      const KeyTypeT& a_featuremap_shape,
                      const KeyTypeT& a_stride,
                      const int a_dim = 3,
                      const std::string a_padding_type = "SAME"
            )
        : m_kernel_ids(a_kernel_ids), m_kernel_weights(a_kernel_weights), m_kernel_shape(a_kernel_shape),
          m_featuremap_shape(a_featuremap_shape), m_stride(a_stride)
    {
        id_in_batch = 0;
        id_in_depth = 1;
        id_in_width = id_in_depth + a_dim - 1;
        id_in_in_channels = id_in_depth + a_dim;
        id_f_depth = 0;
        id_f_width = id_f_depth + a_dim - 1;
        id_f_in_channels = id_f_depth + a_dim;
        id_f_out_channels = id_f_depth + a_dim + 1;

        m_out_shape.resize(m_featuremap_shape.size());
        m_kernel_offset.resize(a_kernel_shape.size());
        for(auto i = 0; i < m_kernel_offset.size(); ++i){
            m_kernel_offset[i] = (a_kernel_shape[i] - 1) / 2; //TODO: precompute filter indices with offset?
            if(i >= id_in_depth && i <= id_in_width){
                m_out_shape[i] = float(a_featuremap_shape[i]) / m_stride[i];
            } else if(i == id_in_in_channels){
                m_out_shape[i] = a_kernel_shape[id_f_out_channels];
            } else {
                m_out_shape[i] = a_featuremap_shape[i];
            }
        }

        //use tensorflows padding pattern
        m_str_padding_offset.assign(a_kernel_shape.size(), 0);
        if(a_padding_type == "SAME"){
            for(uint i = 0; i < m_str_padding_offset.size(); ++i){
                if(int(m_featuremap_shape[i]) % m_stride[i] == 0){
                    m_str_padding_offset[i] = 1;
                }
            }
        }
    }

    void
    computeUpdate(const KeyTypeT& a_id,
                  const ValueT& a_value,
                  std::vector<KeyTypeT>& a_update_id,
                  std::vector<ValueT>& a_update_values
                  ) const {
        a_update_id.resize(m_kernel_ids.size());
        a_update_values.resize(m_kernel_ids.size());
        int valid_count = 0;

        for(uint j = 0; j < m_kernel_ids.size(); ++j){
            if(m_kernel_ids[j][id_f_in_channels] != a_id[id_in_in_channels]) continue; //filter channel != input channel
            bool is_valid = true;
            KeyTypeT update_id(m_featuremap_shape.size(), 0);
            update_id[id_in_batch] = a_id[id_in_batch]; //output channel is filter number
            update_id[id_in_in_channels] = m_kernel_ids[j][id_f_out_channels]; //output channel is filter number
            for(uint k = id_f_depth, l = id_in_depth; k <= id_f_width; ++k, ++l){ //TODO: ugly coding style... prototype
                auto out_plain_id = a_id[l] - m_kernel_ids[j][k] + m_kernel_offset[k];
                if(m_featuremap_shape[l] > 1 && m_stride[l] > 1){
                    if(((out_plain_id ) % m_stride[l]) != m_str_padding_offset[l]){
                        is_valid = false;
                        break;
                    }
                    update_id[l] = float(out_plain_id) / m_stride[l]; //depth, width and height
                } else {
                    update_id[l] = out_plain_id; //depth, width and height
                }
                if(update_id[l] < 0 || update_id[l] >= m_out_shape[l]){    //check boundaries
                    is_valid = false;
                    break;
                }
            }
            if(is_valid){
                auto update_val = a_value * m_kernel_weights[j]; //input value times filter weight at index
                a_update_id[valid_count] = update_id;
                a_update_values[valid_count] = update_val;
                valid_count++;
            }
        }
        a_update_id.resize(valid_count);
        a_update_values.resize(valid_count);
    }

    KeyTypeT
    getOutShape() const {
        return m_out_shape;
    }

private:
    std::vector<KeyTypeT> m_kernel_ids;
    std::vector<int> m_kernel_offset;
    std::vector<WeightT> m_kernel_weights;
    KeyTypeT m_kernel_shape;
    KeyTypeT m_featuremap_shape;
    KeyTypeT m_stride;
    KeyTypeT m_out_shape;
    std::vector<int> m_str_padding_offset;
    int id_in_batch, id_in_depth, id_in_width, id_in_in_channels;
    int id_f_depth, id_f_width, id_f_in_channels, id_f_out_channels;
};
