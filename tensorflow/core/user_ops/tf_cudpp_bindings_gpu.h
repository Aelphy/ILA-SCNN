#pragma once

#include <cudpp_config.h>
#include <cudpp.h>
#include <cudpp_hash.h>
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template<typename T>
CUDPPConfiguration getConfiguration(std::string operand = "add", std::string algorithm = "scan", std::string options = ""){
  CUDPPConfiguration config;
  //set data type
  if(std::is_same<T, int8_t>::value  || std::is_same<T, int8>::value){
    config.datatype = CUDPP_CHAR;
  } else if(std::is_same<T, uint8_t>::value || std::is_same<T, uint8>::value){
    config.datatype = CUDPP_UCHAR;
  } else if(std::is_same<T, int16_t>::value || std::is_same<T, int16>::value){
    config.datatype = CUDPP_SHORT;
  } else if(std::is_same<T, uint16_t>::value || std::is_same<T, uint16>::value){
    config.datatype = CUDPP_USHORT;
  } else if(std::is_same<T, int32_t>::value || std::is_same<T, int32>::value){
    config.datatype = CUDPP_INT;
  } else if(std::is_same<T, uint32_t>::value || std::is_same<T, uint32>::value){
    config.datatype = CUDPP_UINT;
  } else if(std::is_same<T, int64_t>::value || std::is_same<T, int64>::value){
    config.datatype = CUDPP_LONGLONG;
  } else if(std::is_same<T, uint64_t>::value || std::is_same<T, uint64>::value){
    config.datatype = CUDPP_ULONGLONG;
  } else if(std::is_same<T, float>::value){
    config.datatype = CUDPP_FLOAT;
  } else if(std::is_same<T, double>::value){
    config.datatype = CUDPP_DOUBLE;
  } else {
    config.datatype = CUDPP_DATATYPE_INVALID; 
  }

  //set operator
  if(operand == "add"){
    config.op = CUDPP_ADD;
  } else if(operand ==  "multiply"){
    config.op = CUDPP_MULTIPLY;
  } else if(operand ==  "min"){
    config.op = CUDPP_MIN;
  } else if(operand ==  "max"){
    config.op = CUDPP_MAX;
  } else {
    config.op = CUDPP_OPERATOR_INVALID;
  }

  //set algorithm
  if(algorithm == "scan"){
    config.algorithm = CUDPP_SCAN;
  } else if(algorithm ==  "segmented_scan"){
    config.algorithm = CUDPP_SEGMENTED_SCAN;
  } else if(algorithm ==  "compact"){
    config.algorithm = CUDPP_COMPACT;
  } else if(algorithm ==  "reduce"){
    config.algorithm = CUDPP_REDUCE;
  } else if(algorithm ==  "radix sort"){
    config.algorithm = CUDPP_SORT_RADIX;
  } else if(algorithm ==  "merge sort"){
    config.algorithm = CUDPP_SORT_MERGE;
  } else if(algorithm ==  "string sort"){
    config.algorithm = CUDPP_SORT_STRING;
  } else if(algorithm ==  "spmvmult"){
    config.algorithm = CUDPP_SPMVMULT;
  } else if(algorithm ==  "rand md5"){
    config.algorithm = CUDPP_RAND_MD5;
  } else if(algorithm ==  "tridiagonal"){
    config.algorithm = CUDPP_TRIDIAGONAL;
  } else if(algorithm ==  "compress"){
    config.algorithm = CUDPP_COMPRESS;
  } else if(algorithm ==  "listrank"){
    config.algorithm = CUDPP_LISTRANK;
  } else if(algorithm ==  "bwt"){
    config.algorithm = CUDPP_BWT;
  } else if(algorithm ==  "mtf"){
    config.algorithm = CUDPP_MTF;
  } else {
    config.algorithm = CUDPP_ALGORITHM_INVALID;
  } 

  //TODO: options
  config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
  return config;
}

} //namespace tensorflow
