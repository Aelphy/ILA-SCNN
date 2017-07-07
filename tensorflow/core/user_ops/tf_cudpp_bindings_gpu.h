#pragma once

#include <cudpp_config.h>
#include <cudpp.h>
#include <cudpp_hash.h>


template<typename T>
CUDPPConfiguration getConfiguration(std::string operand = "add", std::string algorithm = "scan", std::string options = ""){
  CUDPPConfiguration config;
  //set data type
  switch ((typeid(T)) {
    case typeid(int8_t):
      config.datatype = CUDPP_CHAR;
      break;
    case typeid(uint8_t):
      config.datatype = CUDPP_UCHAR;
      break;
    case typeid(int16_t):
      config.datatype = CUDPP_SHORT;
      break;
    case typeid(uint16_t):
      config.datatype = CUDPP_USHORT;
      break;
    case typeid(int32_t):
      config.datatype = CUDPP_INT;
      break;
    case typeid(uint32_t):
      config.datatype = CUDPP_UINT;
      break;
    case typeid(int64_t):
      config.datatype = CUDPP_LONGLONG;
      break;
    case typeid(uint64_t):
      config.datatype = CUDPP_ULONGLONG;
      break;
    case typeid(float):
      config.datatype = CUDPP_FLOAT;
      break;
    case typeid(double):
      config.datatype = CUDPP_DOUBLE;
      break;
    default:
      config.datatype = CUDPP_DATATYPE_INVALID 
  }

  //set operator
  switch(operand){
    case "add":
      config.op = CUDPP_ADD;
      break;
    case "multiply":
      config.op = CUDPP_MULTIPLY;
      break;
    case "min":
      config.op = CUDPP_MIN;
      break;
    case "max":
      config.op = CUDPP_MAX;
      break;
    default:
      config.op = CUDPP_OPERATOR_INVALID;
  }
  switch(algorithm):
    case "scan":
      config.algorithm = CUDPP_SCAN;
      break;
    case "segmented_scan":
      config.algorithm = CUDPP_SEGMENTED_SCAN;
      break;
    case "compact":
      config.algorithm = CUDPP_COMPACT;
      break;
    case "reduce":
      config.algorithm = CUDPP_REDUCE;
      break;
    case "radix sort":
      config.algorithm = CUDPP_SORT_RADIX;
      break;
    case "merge sort":
      config.algorithm = CUDPP_SORT_MERGE;
      break;
    case "string sort":
      config.algorithm = CUDPP_SORT_STRING;
      break;
    case "spmvmult":
      config.algorithm = CUDPP_SPMVMULT;
      break;
    case "rand":
      config.algorithm = CUDPP_RAND_MD5;
      break;
    case "tridiagonal":
      config.algorithm = CUDPP_TRIDIAGONAL;
      break;
    case "compress":
      config.algorithm = CUDPP_COMPRESS;
      break;
    case "listrank":
      config.algorithm = CUDPP_LISTRANK;
      break;
    case "bwt":
      config.algorithm = CUDPP_BWT;
      break;
    case "mtf":
      config.algorithm = CUDPP_MTF;
      break;
    default:
      config.algorithm = CUDPP_ALGORITHM_INVALID;
  } 
  //TODO: options
  //config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
  return config;
}
