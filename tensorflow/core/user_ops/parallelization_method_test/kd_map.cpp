#include <iostream>
#include <map>
#include <omp.h>
#include "time/time.h"
#include "indexing.h"
#include "rnd_matrix.h"
#include "vote3d_conv.h"
#include "hash_map.h"

int main(){
    typedef u_int64_t IndexT;
    typedef int ValueT;
    typedef std::vector<IndexT> KeyTypeT;
    RandomTensor<IndexT, ValueT, KeyTypeT > rnd;

    //sparse feature map
    KeyTypeT shape; shape.push_back(1); shape.push_back(50); shape.push_back(50); shape.push_back(50); shape.push_back(1);
    std::vector<KeyTypeT> sparse_feature_ids;
    std::vector<ValueT> sparse_feature_vals;
    rnd.createTensor(shape, 0.5, sparse_feature_ids, sparse_feature_vals);

    //sparse kernel
    KeyTypeT filter_shape; filter_shape.push_back(3); filter_shape.push_back(3); filter_shape.push_back(3); filter_shape.push_back(1); filter_shape.push_back(1);
    std::vector<KeyTypeT> sparse_kernel_ids;
    std::vector<ValueT> sparse_kernel_vals;
    rnd.createTensor(filter_shape, 0.8, sparse_kernel_ids, sparse_kernel_vals);


    t1h::Timer t;

    LFMap<KeyTypeT, ValueT> map(shape);
    LFMap<KeyTypeT, ValueT> *map_ptr = &map;

    std::vector<IndexT> stride; stride.push_back(1); stride.push_back(1); stride.push_back(1); stride.push_back(1); stride.push_back(1);

    SparseConvUpdate<KeyTypeT, ValueT, ValueT> scu(sparse_kernel_ids,
                                                   sparse_kernel_vals,
                                                   filter_shape,
                                                   shape,
                                                   stride);
    SparseConvUpdate<KeyTypeT, ValueT, ValueT> *scu_ptr = &scu;

    t.tic();
#pragma omp parallel for firstprivate(map_ptr, scu_ptr, sparse_feature_ids, sparse_feature_vals, sparse_kernel_ids, sparse_kernel_vals)
    for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
        //"sparse convolution":
        std::vector<KeyTypeT> conv_update_idx;
        std::vector<ValueT> conv_update_val;
        scu_ptr->computeUpdate(sparse_feature_ids[idx], sparse_feature_vals[idx], conv_update_idx, conv_update_val);

#pragma omp parallel for
        for(uint idx2 = 0; idx2 < conv_update_idx.size(); ++idx2)
            map_ptr->update(conv_update_idx[idx2], conv_update_val[idx2]);
    }
    std::cout << "time parallel " << t.tac() << std::endl;
    std::vector<KeyTypeT > res_ids;
    std::vector<ValueT> res_vals;
    map.traverse(res_ids, res_vals);



    t.tic();
    std::map<KeyTypeT, ValueT> cmp_map;
    for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
        std::vector<KeyTypeT> conv_update_idx;
        std::vector<ValueT> conv_update_val;
        scu_ptr->computeUpdate(sparse_feature_ids[idx], sparse_feature_vals[idx], conv_update_idx, conv_update_val);
        for(uint idx2 = 0; idx2 < conv_update_idx.size(); ++idx2){
            auto &id = conv_update_idx[idx2];
            auto &val = conv_update_val[idx2];
            std::pair<std::map<KeyTypeT, ValueT>::iterator, bool> ret = cmp_map.insert(std::make_pair(id, val));
            if(!ret.second){
                ret.first->second += val;
            }
        }
    }
    std::cout << "time serial " << t.tac()<< std::endl;


    auto cnti = 0;
    for(auto it = cmp_map.begin(); it != cmp_map.end(); ++it, ++cnti){
        auto vals = it->second;
        auto ids = it->first;
        if(vals == map.getValue(ids)) continue;
        std::cout << "ERROR: expected " << vals <<" got: " << res_vals[cnti] << " at: ";
        for(size_t j = 0; j < ids.size(); ++j){
            std::cout << " " << ids[j];
        }
        std::cout << std::endl;
    }

    return 0;
}
