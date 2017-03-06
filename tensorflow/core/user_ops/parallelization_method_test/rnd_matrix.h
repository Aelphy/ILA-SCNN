#pragma once

#include <time.h>
#include <vector>
#include "indexing.h"

template<typename IndexT, typename ValueT, typename KeyType>
class RandomTensor {
public:
  RandomTensor(const ValueT a_max_rnd_val = 10) : m_max_rnd_val(a_max_rnd_val){
    srand (time(NULL));
  }

  void
  createTensor( const KeyType& a_shape,
                const float a_density,
                std::vector<KeyType>& a_indices,
                std::vector<ValueT>& a_values) const {

      int total_size = 1;
      for(int i = 0; i < (int) a_shape.size(); ++i){
        total_size = total_size * a_shape[i]; 
      }
      
      int reduced_size = total_size * a_density;
      a_values.resize(reduced_size);
      a_indices.resize(reduced_size);
      for(auto i = 0; i < reduced_size; ++i){
        auto rdn_idx = rand() % (int) reduced_size; //TODO: unique picks
        a_indices[i] = getHighDimIndexVec<IndexT, KeyType>(rdn_idx, a_shape);
        a_values[i] = ValueT(rand() % (int) m_max_rnd_val); //TODO: unique picks
      }
  }
protected:
  ValueT m_max_rnd_val;
};
