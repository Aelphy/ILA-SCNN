#include <iostream>
#include <map>
#include <omp.h>
#include <parallel/algorithm>
#include "time/time.h"
#include "indexing.h"
#include "rnd_matrix.h"
#include "vote3d_conv.h"
#include "hash_map.h"
#include "merge_sort.h"
#include "generic_matrix_txt_io.h"

template<typename T> bool cmp_pair_eq (const T& i, const T& j) {
  return (i.first == j.first);
}

int main(){
    const int max_num_threads = 8;
    const int num_trials = 10;
    typedef u_int64_t IndexT;
    typedef int ValueT;
    typedef std::vector<IndexT> KeyTypeT;
    RandomTensor<IndexT, ValueT, KeyTypeT > rnd;

    //sparse feature map
    KeyTypeT shape; shape.push_back(1); shape.push_back(100); shape.push_back(100); shape.push_back(100); shape.push_back(1);
    std::vector<KeyTypeT> sparse_feature_ids;
    std::vector<KeyTypeT>* sparse_feature_ids_ptr = &sparse_feature_ids;
    std::vector<ValueT> sparse_feature_vals;
    std::vector<ValueT>* sparse_feature_vals_ptr = &sparse_feature_vals;
    rnd.createTensor(shape, 0.1, sparse_feature_ids, sparse_feature_vals);

    //sparse kernel
    KeyTypeT filter_shape; filter_shape.push_back(3); filter_shape.push_back(3); filter_shape.push_back(3); filter_shape.push_back(1); filter_shape.push_back(1);
    std::vector<KeyTypeT> sparse_kernel_ids;
    std::vector<ValueT> sparse_kernel_vals;
    rnd.createTensor(filter_shape, 0.9, sparse_kernel_ids, sparse_kernel_vals);


    std::vector<IndexT> stride; stride.push_back(1); stride.push_back(1); stride.push_back(1); stride.push_back(1); stride.push_back(1);

    SparseConvUpdate<KeyTypeT, ValueT, ValueT> scu(sparse_kernel_ids,
                                                   sparse_kernel_vals,
                                                   filter_shape,
                                                   shape,
                                                   stride);
    SparseConvUpdate<KeyTypeT, ValueT, ValueT> *scu_ptr = &scu;

    t1h::Timer t;

    LFMap<KeyTypeT, ValueT> map(shape);
    LFMap<KeyTypeT, ValueT> *map_ptr = &map;


//    std::vector<float> res_hash_map(max_num_threads, 0);
//    for(size_t num = 1; num <= max_num_threads; ++num){
//        for(size_t trial = 0; trial < num_trials; ++trial){
//            map.m_cmap.clear();
//            t.tic();
//            map.m_cmap.reserve(sparse_feature_vals.size() * sparse_kernel_vals.size());
//#pragma omp parallel for firstprivate(map_ptr, scu_ptr, sparse_feature_ids_ptr, sparse_feature_vals_ptr) num_threads(num)
//            for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
//                //"sparse convolution":
//                std::vector<KeyTypeT> conv_update_idx;
//                std::vector<ValueT> conv_update_val;
//                scu_ptr->computeUpdate(sparse_feature_ids_ptr->at(idx), sparse_feature_vals_ptr->at(idx), conv_update_idx, conv_update_val);
//                for(uint idx2 = 0; idx2 < conv_update_idx.size(); ++idx2)
//                    map_ptr->update(conv_update_idx[idx2], conv_update_val[idx2]);
//            }
//            res_hash_map[num - 1] += t.tac();
//        }
//        res_hash_map[num - 1] = res_hash_map[num - 1] / num_trials;
//        std::cout << "hash map, time parallel: " << res_hash_map[num - 1] << " num  threads: " << num << std::endl;
//    }
//    pcc::writeTXTFile("result_hash_map.txt", res_hash_map, res_hash_map.size(), 1);
//    std::vector<KeyTypeT > res_ids;
//    std::vector<ValueT> res_vals;
//    map.traverse(res_ids, res_vals);



//    std::map<IndexT, ValueT> cmp_map;
//    std::vector<float> t_result_map(1,0);
//    for(size_t trial = 0; trial < num_trials; ++trial){
//        cmp_map.clear();
//        t.tic();
//        for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
//            std::vector<KeyTypeT> conv_update_idx;
//            std::vector<ValueT> conv_update_val;
//            scu_ptr->computeUpdate(sparse_feature_ids[idx], sparse_feature_vals[idx], conv_update_idx, conv_update_val);
//            for(uint idx2 = 0; idx2 < conv_update_idx.size(); ++idx2){
//                IndexT id = getIndex1D<IndexT, KeyTypeT>(conv_update_idx[idx2], scu_ptr->getOutShape());
//                auto &val = conv_update_val[idx2];
//                std::pair<std::map<IndexT, ValueT>::iterator, bool> ret = cmp_map.insert(std::make_pair(id, val));
//                if(!ret.second){
//                    ret.first->second += val;
//                }
//            }
//        }
//        t_result_map[0] += t.tac();
//    }
//    t_result_map[0] = t_result_map[0] / num_trials;
//    std::cout << "time serial " << t_result_map[0] << std::endl;
//    pcc::writeTXTFile("result_map.txt", t_result_map, t_result_map.size(), 1);


//    auto cnti = 0;
//    for(auto it = cmp_map.begin(); it != cmp_map.end(); ++it, ++cnti){
//        auto vals = it->second;
//        auto ids = it->first;
//        if(vals == map.getValue(ids)) continue;
//        std::cout << "ERROR 1: expected " << vals <<" got: " << res_vals[cnti] << " at: " << ids;
//        std::cout << std::endl;
//    }
    std::vector<IndexT> all_keys_1d, *all_keys_1d_ptr = &all_keys_1d;
    std::vector<ValueT> all_values, *all_values_ptr = &all_values;
    std::vector<std::pair<IndexT,ValueT> > all_pairs, *all_pairs_ptr = &all_pairs;
//    std::vector<float> res_merge_sort(max_num_threads, 0);
//    for(size_t num = 1; num <= max_num_threads; ++num){
//        for(size_t trial = 0; trial < num_trials; ++trial){
//            all_keys_1d.clear();
//            all_values.clear();
//            all_pairs.clear();
//            t.tic();

//#pragma omp parallel for firstprivate(scu_ptr, sparse_feature_ids_ptr, sparse_feature_vals_ptr, all_values_ptr, all_pairs_ptr, all_keys_1d_ptr)
//            for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
//                std::vector<KeyTypeT> conv_update_idx;
//                std::vector<ValueT> conv_update_val;
//                scu_ptr->computeUpdate(sparse_feature_ids[idx], sparse_feature_vals[idx], conv_update_idx, conv_update_val);

//                std::vector<std::pair<IndexT, ValueT> > conv_update_idx_1d(conv_update_idx.size());
//                for(auto idx2 = 0; idx2 < conv_update_idx.size(); ++idx2){
//                    conv_update_idx_1d[idx2] = std::make_pair(getIndex1D<IndexT, KeyTypeT>(conv_update_idx[idx2], scu_ptr->getOutShape()), conv_update_val[idx2]);
//                }
//#pragma omp critical
//                {
//                    all_pairs_ptr->insert(all_pairs_ptr->end(), conv_update_idx_1d.begin(), conv_update_idx_1d.end());
//                }
//            }
//            //merge_join_sort(all_keys_1d, all_values);
//            omp_set_num_threads(num);
//            __gnu_parallel::sort(all_pairs.begin(), all_pairs.end());
//            std::vector<std::pair<IndexT,ValueT> > joined_pairs(all_pairs.size()), *joined_pairs_ptr = &joined_pairs;
//            int cnt = 0;
////#pragma omp parallel for firstprivate(all_pairs_ptr, joined_pairs_ptr) shared(cnt)
//            for(int i = 1; i < all_pairs_ptr->size(); ++i){
//                if(all_pairs_ptr->at(i - 1).first != all_pairs_ptr->at(i).first  || i + 1 == all_pairs_ptr->size()){
//                    auto this_count = cnt++;
//                    int id1 = i - 2;
//                    int id2 = i - 1;
//                    if(i + 1 == all_pairs_ptr->size()){
//                        id1 = i - 1;
//                        id2 = i;
//                    }
//                    auto val = all_pairs_ptr->at(id2).second;
//                    while(id1 >= 0 && all_pairs_ptr->at(id2).first == all_pairs_ptr->at(id1).first){
//                        val += all_pairs_ptr->at(id1).second;
//                        --id1;
//                    }
//                    joined_pairs_ptr->at(this_count) = all_pairs_ptr->at(id2);
//                    joined_pairs_ptr->at(this_count).second = val;
//                }
//            }
//            joined_pairs.resize(cnt);
//            all_keys_1d.resize(cnt);
//            all_values.resize(cnt);
//#pragma omp parallel for firstprivate(all_keys_1d_ptr, all_values_ptr, joined_pairs_ptr)
//            for(int i = 0; i < joined_pairs_ptr->size(); ++i){
//                all_keys_1d_ptr->at(i) = joined_pairs_ptr->at(i).first;
//                all_values_ptr->at(i) = joined_pairs_ptr->at(i).second;
//            }


//            res_merge_sort[num - 1] += t.tac();
//        }
//        res_merge_sort[num - 1] = res_merge_sort[num - 1] / num_trials;
//        std::cout << "join-merge-sort, time parallel: " << res_merge_sort[num - 1] << " num  threads: " << num << std::endl;
//    }
//    pcc::writeTXTFile("result_merge_sort.txt", res_merge_sort, res_merge_sort.size(), 1);

//    for(auto i = 0; i < all_values.size(); ++i){
//        auto vals = all_values[i];
//        auto ids = getHighDimIndexVec(all_keys_1d[i], scu_ptr->getOutShape());
//        auto it = cmp_map.find(all_keys_1d[i]);
//        if(vals == it->second) continue;
//        std::cout << "ERROR2: expected " << vals <<" got: " << it->second << " at: " << all_keys_1d[i] << "  ";
//        for(size_t j = 0; j < ids.size(); ++j){
//            std::cout << " " << ids[j];
//        }
//        std::cout << std::endl;
//    }

//    int nr_steps = 20;
//    std::vector<float> sparse_kernel_test_merge_sort(nr_steps + 1, 0);
//    for(size_t num = 0; num <= nr_steps; ++num){
//        rnd.createTensor(filter_shape, 1. - float(num) / nr_steps, sparse_kernel_ids, sparse_kernel_vals);
//        SparseConvUpdate<KeyTypeT, ValueT, ValueT> scu1(sparse_kernel_ids,
//                                                       sparse_kernel_vals,
//                                                       filter_shape,
//                                                       shape,
//                                                       stride);
//        SparseConvUpdate<KeyTypeT, ValueT, ValueT> *scu_ptr = &scu1;

//        for(size_t trial = 0; trial < num_trials; ++trial){
//            map.m_cmap.clear();
//            t.tic();
//            map.m_cmap.reserve(sparse_feature_vals.size() * sparse_kernel_vals.size());
//#pragma omp parallel for firstprivate(map_ptr, scu_ptr, sparse_feature_ids_ptr, sparse_feature_vals_ptr) num_threads(8)
//            for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
//                //"sparse convolution":
//                std::vector<KeyTypeT> conv_update_idx;
//                std::vector<ValueT> conv_update_val;
//                scu_ptr->computeUpdate(sparse_feature_ids_ptr->at(idx), sparse_feature_vals_ptr->at(idx), conv_update_idx, conv_update_val);
//                for(uint idx2 = 0; idx2 < conv_update_idx.size(); ++idx2)
//                    map_ptr->update(conv_update_idx[idx2], conv_update_val[idx2]);
//            }
//            sparse_kernel_test_merge_sort[num] += t.tac();
//        }
//        sparse_kernel_test_merge_sort[num] = sparse_kernel_test_merge_sort[num] / num_trials;
//        std::cout << "join-merge-sort, time parallel: " << sparse_kernel_test_merge_sort[num] << " num  threads: " << num << " number filter weights: " << sparse_kernel_vals.size() << std::endl;
//    }
//    pcc::writeTXTFile("result_sparse_kernel_merge_sort.txt", sparse_kernel_test_merge_sort, sparse_kernel_test_merge_sort.size(), 1);



    int nr_steps = 20;
    std::vector<float> sparse_kernel_test_merge_sort(nr_steps + 1, 0);
    for(size_t num = 0; num <= nr_steps; ++num){
        rnd.createTensor(filter_shape, 1. - float(num) / nr_steps, sparse_kernel_ids, sparse_kernel_vals);
        SparseConvUpdate<KeyTypeT, ValueT, ValueT> scu1(sparse_kernel_ids,
                                                       sparse_kernel_vals,
                                                       filter_shape,
                                                       shape,
                                                       stride);
        SparseConvUpdate<KeyTypeT, ValueT, ValueT> *scu_ptr = &scu1;

        for(size_t trial = 0; trial < num_trials; ++trial){
            all_keys_1d.clear();
            all_values.clear();
            all_pairs.clear();
            t.tic();

#pragma omp parallel for firstprivate(scu_ptr, sparse_feature_ids_ptr, sparse_feature_vals_ptr, all_values_ptr, all_pairs_ptr, all_keys_1d_ptr)
            for(uint idx = 0; idx < sparse_feature_vals.size(); ++idx){
                std::vector<KeyTypeT> conv_update_idx;
                std::vector<ValueT> conv_update_val;
                scu_ptr->computeUpdate(sparse_feature_ids[idx], sparse_feature_vals[idx], conv_update_idx, conv_update_val);

                std::vector<std::pair<IndexT, ValueT> > conv_update_idx_1d(conv_update_idx.size());
                for(auto idx2 = 0; idx2 < conv_update_idx.size(); ++idx2){
                    conv_update_idx_1d[idx2] = std::make_pair(getIndex1D<IndexT, KeyTypeT>(conv_update_idx[idx2], scu_ptr->getOutShape()), conv_update_val[idx2]);
                }
#pragma omp critical
                {
                    all_pairs_ptr->insert(all_pairs_ptr->end(), conv_update_idx_1d.begin(), conv_update_idx_1d.end());
                }
            }
            //merge_join_sort(all_keys_1d, all_values);
            omp_set_num_threads(8);
            __gnu_parallel::sort(all_pairs.begin(), all_pairs.end());
            std::vector<std::pair<IndexT,ValueT> > joined_pairs(all_pairs.size()), *joined_pairs_ptr = &joined_pairs;
            int cnt = 0;
//#pragma omp parallel for firstprivate(all_pairs_ptr, joined_pairs_ptr) shared(cnt)
            for(int i = 1; i < all_pairs_ptr->size(); ++i){
                if(all_pairs_ptr->at(i - 1).first != all_pairs_ptr->at(i).first  || i + 1 == all_pairs_ptr->size()){
                    auto this_count = cnt++;
                    int id1 = i - 2;
                    int id2 = i - 1;
                    if(i + 1 == all_pairs_ptr->size()){
                        id1 = i - 1;
                        id2 = i;
                    }
                    auto val = all_pairs_ptr->at(id2).second;
                    while(id1 >= 0 && all_pairs_ptr->at(id2).first == all_pairs_ptr->at(id1).first){
                        val += all_pairs_ptr->at(id1).second;
                        --id1;
                    }
                    joined_pairs_ptr->at(this_count) = all_pairs_ptr->at(id2);
                    joined_pairs_ptr->at(this_count).second = val;
                }
            }
            joined_pairs.resize(cnt);
            all_keys_1d.resize(cnt);
            all_values.resize(cnt);
#pragma omp parallel for firstprivate(all_keys_1d_ptr, all_values_ptr, joined_pairs_ptr)
            for(int i = 0; i < joined_pairs_ptr->size(); ++i){
                all_keys_1d_ptr->at(i) = joined_pairs_ptr->at(i).first;
                all_values_ptr->at(i) = joined_pairs_ptr->at(i).second;
            }


            sparse_kernel_test_merge_sort[num] += t.tac();
        }
        sparse_kernel_test_merge_sort[num] = sparse_kernel_test_merge_sort[num] / num_trials;
        std::cout << "join-merge-sort, time parallel: " << sparse_kernel_test_merge_sort[num] << " num  threads: " << num << " number filter weights: " << sparse_kernel_vals.size() << std::endl;
    }
    pcc::writeTXTFile("result_sparse_kernel_merge_sort.txt", sparse_kernel_test_merge_sort, sparse_kernel_test_merge_sort.size(), 1);




    return 0;
}
