#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <cmath>

//Problem! memory: initially needs to store all! additions in RAM at once. Only slowly reduces number.

template<typename KeyT, typename ValueT, class Alloc, template<typename, class Alloc> class ContainerT, template<typename> class CompLessT>
int merge_join(ContainerT<KeyT, Alloc>& l, ContainerT<KeyT, Alloc> vl,
               ContainerT<KeyT, Alloc>& r, ContainerT<KeyT, Alloc>& vr,
               ContainerT<KeyT, Alloc>& o, ContainerT<KeyT, Alloc>& vo){
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


template<typename KeyT, typename ValueT, class Alloc, template<typename, class Alloc> class ContainerT, template<typename> class CompLessT = std::less_equal>
void merge_join_sort(ContainerT<KeyT, Alloc>& keys, ContainerT<ValueT, Alloc>& values){
    //devide
    int num_entries(keys.size());
    ContainerT<ContainerT<KeyT, Alloc>,  Alloc> devided_keys(num_entries);
    ContainerT<ContainerT<ValueT, Alloc>, Alloc> devided_values(num_entries);
    for(int i = 0; i < keys.size(); ++i){
        devided_keys[i] = ContainerT<KeyT, Alloc>(std::make_move_iterator(keys.begin() + i),
                           std::make_move_iterator(keys.begin() + i + 1));
        devided_values[i] = ContainerT<KeyT, Alloc>(std::make_move_iterator(values.begin() + i),
                           std::make_move_iterator(values.begin() + i + 1));
    }


    //conquer
    while(devided_keys.size() > 1){
        ContainerT<ContainerT<KeyT, Alloc>,  Alloc> merged_keys(ceil(devided_keys.size() / 2.));
        ContainerT<ContainerT<ValueT, Alloc>, Alloc> merged_values(ceil(devided_keys.size() / 2.));
        for(int j = 0, k = 0; j < devided_keys.size(); j += 2, ++k){
            if(j + 1 < devided_keys.size()){
                merge_join<KeyT, ValueT, Alloc, ContainerT, CompLessT>(
                                devided_keys[j], devided_values[j],
                                devided_keys[j+1], devided_values[j+1],
                                merged_keys[k], merged_values[k]
                        );
            } else {
                merged_keys[k] = devided_keys[j];
                merged_values[k] = devided_values[j];
            }
        }
        devided_keys = merged_keys;
        devided_values = merged_values;
    }


    keys = devided_keys[0];
    values = devided_values[0];
}

