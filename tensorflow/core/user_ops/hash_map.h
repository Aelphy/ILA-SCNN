#pragma once

#include <libcuckoo/cuckoohash_map.hh>
#include <libcuckoo/city_hasher.hh>

template<typename KeyType, typename ValueT>
class LFMap {
public:
    typedef uint64_t IndexT;
    typedef cuckoohash_map<IndexT, ValueT, CityHasher<IndexT> > ConcurrentMap;

    LFMap(KeyType a_shape) : m_shape(a_shape){}

    void
    update(KeyType a_key, ValueT a_v){
        auto updatefn = [a_v](ValueT& num) { num+=a_v; };
        IndexT hash_key = getIndex1D(a_key, m_shape);
        m_cmap.upsert(hash_key, updatefn, a_v);
    }

    void
    insert(KeyType a_key, ValueT a_v){
        IndexT hash_key = getIndex1D(a_key, m_shape);
        m_cmap.insert(hash_key, a_v);
    }

    void
    traverse(std::vector<KeyType>& keys, std::vector<ValueT>& values){
        keys.resize(m_cmap.size());
        values.resize(m_cmap.size());
        auto lmap = m_cmap.lock_table();
        int cnt = 0;
        for(auto it = lmap.begin(); it != lmap.end(); ++it, cnt++){
            keys[cnt] = getHighDimIndexVec(it->first, m_shape);
            values[cnt] = it->second;
        }
    }

    bool
    find(const KeyType& a_key, ValueT& a_val) const {
        IndexT hash_key = getIndex1D(a_key, m_shape);
        return m_cmap.find(hash_key, a_val);
    }

    ValueT
    getValue(const KeyType& a_key) const {
        IndexT hash_key = getIndex1D(a_key, m_shape);
        return m_cmap.find(hash_key);
    }

    ValueT
    getValue(const IndexT& a_key_1D) const {
        return m_cmap.find(a_key_1D);
    }

    void
    reserve(size_t size){
      m_cmap.reserve(size);
    }

    ConcurrentMap m_cmap;
protected:
    inline KeyType
    getHighDimIndexVec(IndexT a_index, KeyType a_shape) const
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
        for(size_t i = 0; i < index.size(); ++i) assert(index[i] >= 0 && index[i] < a_shape[i]);
        return index;
    }


    inline IndexT
    getIndex1D(KeyType a_index, KeyType a_shape) const
    {
        assert(a_index.size() == a_shape.size());
        IndexT index_1d = 0;
        IndexT mult = 1;
        for(size_t i = 0; i < a_shape.size(); ++i){
            index_1d += a_index[i] * mult;
            mult = mult * a_shape[i];
        }
        return index_1d;
    }
    KeyType m_shape;
};
