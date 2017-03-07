#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <omp.h>

//Problem! memory: initially needs to store all! additions in RAM at once. Only slowly reduces number.

template<typename KeyT, typename ValueT, class Alloc1, class Alloc2, template<typename, class> class ContainerT, template<typename> class CompLessT>
int merge_join(ContainerT<KeyT, Alloc1>& l, ContainerT<ValueT, Alloc2> vl,
               ContainerT<KeyT, Alloc1>& r, ContainerT<ValueT, Alloc2>& vr,
               ContainerT<KeyT, Alloc1>& o, ContainerT<ValueT, Alloc2>& vo){
    o.resize(l.size() + r.size());
    vo.resize(l.size() + r.size());

    CompLessT<KeyT> cmp_less;
    int i = 0, j = 0, k = 0;
    for(; i < l.size() && j < r.size(); ++k){
        if(l[i] == r[j]){
            o[k] = l[i];
            vo[k] = vl[i] + vr[j];
            i++; j++;
        } else if(cmp_less(l[i],r[j])){
            o[k] = l[i];
            vo[k] = vl[i];
            i++;
        } else {
            o[k] = r[j];
            vo[k] = vr[j];
            j++;
        }
    }

    for(; i < l.size(); ++i, ++k){
        o[k] = l[i];
        vo[k] = vl[i];
    }

    for(; j < r.size(); ++j, ++k){
        o[k] = r[j];
        vo[k] = vr[j];
    }

    o.resize(k);
    vo.resize(k);
}


template<typename KeyT, typename ValueT, class Alloc1, class Alloc2, template<typename, class> class ContainerT, template<typename> class CompLessT = std::less_equal>
void merge_join_sort(ContainerT<KeyT, Alloc1>& keys, ContainerT<ValueT, Alloc2>& values){



//    //divide
//    int num_entries(keys.size());
//    ContainerT<ContainerT<KeyT, Alloc1>,  Alloc1> divided_keys(num_entries);
//    ContainerT<ContainerT<ValueT, Alloc2>, Alloc1> divided_values(num_entries);
//    for(int i = 0; i < keys.size(); ++i){
//        divided_keys[i] = ContainerT<KeyT, Alloc1>(std::make_move_iterator(keys.begin() + i),
//                           std::make_move_iterator(keys.begin() + i + 1));
//        divided_values[i] = ContainerT<ValueT, Alloc2>(std::make_move_iterator(values.begin() + i),
//                           std::make_move_iterator(values.begin() + i + 1));
//    }


//    //conquer
//    while(divided_keys.size() > 1){
//        ContainerT<ContainerT<KeyT, Alloc1>,  Alloc1> merged_keys(ceil(divided_keys.size() / 2.));
//        ContainerT<ContainerT<ValueT, Alloc2>, Alloc1> merged_values(ceil(divided_keys.size() / 2.));
//        ContainerT<ContainerT<KeyT, Alloc1>,  Alloc1>* merged_keys_ptr = &merged_keys;
//        ContainerT<ContainerT<ValueT, Alloc2>, Alloc1>* merged_values_ptr = &merged_values;
//        ContainerT<ContainerT<KeyT, Alloc1>,  Alloc1>* divided_keys_ptr = &divided_keys;
//        ContainerT<ContainerT<ValueT, Alloc2>, Alloc1>* divided_values_ptr = &divided_values;

//#pragma omp parallel for firstprivate(merged_keys_ptr, merged_values_ptr, divided_keys_ptr, divided_values_ptr)
//        for(int j = 0; j < divided_keys_ptr->size(); j += 2){
//            int k = j / 2;
//            if(j + 1 < divided_keys_ptr->size()){
//                merge_join<KeyT, ValueT, Alloc1, Alloc2, ContainerT, CompLessT>(
//                                (*divided_keys_ptr)[j], (*divided_values_ptr)[j],
//                                (*divided_keys_ptr)[j+1], (*divided_values_ptr)[j+1],
//                                (*merged_keys_ptr)[k], (*merged_values_ptr)[k]
//                        );
//            } else {
//                (*merged_keys_ptr)[k] = (*divided_keys_ptr)[j];
//                (*merged_values_ptr)[k] = (*divided_values_ptr)[j];
//            }
//        }
//        divided_keys = merged_keys;
//        divided_values = merged_values;
//    }
//    keys = divided_keys[0];
//    values = divided_values[0];
}

