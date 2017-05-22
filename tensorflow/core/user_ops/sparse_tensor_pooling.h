#pragma once

#include <omp.h>
#include "hash_map.h"

namespace tensorflow {
  template<typename Device, typename T, typename IndexT, typename ValueT, typename ShapeT, typename IndiceT>
  inline void max_pooling(IndexT& in_ind,
                      ValueT& in_val, 
                      ShapeT& in_sh,
                      ShapeT& p_sh,
                      std::vector<std::vector<IndiceT> >& output_ids,
                      std::vector<T>& output_vals,
                      std::vector<IndiceT>& output_shape,
                      std::vector<IndiceT>& corresponding_output_ids,
                      int& dim)
  {
    output_shape.resize(in_sh.dimension(0));
    for(size_t i = 0; i < output_shape.size(); ++i){
      output_shape[i] = std::max<IndiceT>(1, ceil(float(in_sh(i)) / p_sh(i)));
    }
    LFMap<std::vector<IndiceT>, std::pair<T, IndiceT> > map(output_shape);
    map.reserve(in_ind.dimension(0));

    auto map_ptr = &map; auto in_ind_ptr = &in_ind; auto p_sh_ptr = &p_sh; auto in_val_ptr = &in_val;

#pragma omp parallel for firstprivate(in_ind_ptr, p_sh_ptr, in_val_ptr, map_ptr)
    for(size_t i = 0; i < (*in_ind_ptr).dimension(0); ++i){
      std::vector<IndiceT> update_id((*in_ind_ptr).dimension(1), 0);
      for(size_t j = 0; j < update_id.size(); ++j){
        update_id[j] = (*in_ind_ptr)(i, j) / (*p_sh_ptr)(j);
      }
      map_ptr->update_max(update_id, std::make_pair((*in_val_ptr)(i), i));
    }

    map_ptr->traverse(output_ids, output_vals, corresponding_output_ids);
  }
}
