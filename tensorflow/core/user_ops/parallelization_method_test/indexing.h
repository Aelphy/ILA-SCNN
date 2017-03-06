#pragma once

template<typename IndexT, typename KeyType> KeyType
getHighDimIndexVec(IndexT a_index, KeyType a_shape)
{
  KeyType fact(a_shape.size());
  if(a_shape.size() > 0) fact[0] = 1;
  for(size_t i = 1; i < a_shape.size(); ++i){
      fact[i] = a_shape[i - 1] * fact[i - 1];
  }
  KeyType index(a_shape.size());
  IndexT r = a_index;
  for(size_t i = 0; i < a_shape.size(); ++i){
      index[a_shape.size() - i - 1] = r / fact[a_shape.size() - i - 1];
      r = r % fact[a_shape.size() - i - 1];
  }
  return index;
}


template<typename IndexT, typename KeyType> IndexT
getIndex1D(KeyType a_index, KeyType a_shape)
{
  IndexT index_1d = 0;
  IndexT mult = 1;
  for(size_t i = 0; i < a_shape.size(); ++i){
      index_1d += a_index[i] * mult;
      mult = mult * a_shape[i];
  }
  return index_1d;
}


