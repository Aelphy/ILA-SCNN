import tensorflow as tf
import random
import numpy as np
import time

def dense_to_sparse(dense, shape):
  ind = []
  val = []
  sh = shape
  for index, value in np.ndenumerate(dense):
    if value == 0:
      continue
    ind.append(index)
    val.append(value)
  indice = np.array(ind, dtype=np.int64)
  values = np.array(val, dtype=np.float32)
  assert(np.all(sparse_to_dense(indice, values, shape).flat() - dense.flat == 0))
  return [indice, values, sh]

def sparse_to_dense(ind, val, shape):
  dense = np.zeros(shape, dtype=np.float32)
  for idx in range(0, len(ind)):
    ind_helper = [slice(None)]*len(ind[idx])
    dense[tuple(ind[idx])] = val[idx]
  return dense

def idkD_to_id1D(idx, shape):
   index_1d = 0;
   mult = 1;
   for i in range(0,len(shape)):
     index_1d += idx[i] * mult;
     mult = mult * shape[i];
   assert(np.all(id1D_to_idkD(index_1d, shape) - idx == 0))
   return index_1d

def id1D_to_idkD(inid, shape):
  fact = []
  dim = 0 
  fact.append(1);
  lastdim=0
  for d in shape:
    if dim > 0:
      fact.append(fact[dim - 1] * lastdim)
    lastdim = d
    dim += 1

  r = int(inid)
  idx = 0
  idkd = fact
  for d in shape:
    denum = int(fact[dim - idx - 1])
    idkd[dim - idx - 1] = int(float(r) / float(denum))
    r = int(r % denum)
    idx += 1
  rt =  np.array(idkd, dtype=np.int64)
  return rt

def createRandomSparseTensor(non_zero_percentage, shape, min_range = 1, max_range = 10):
  full_range = max_range - min_range
  random.seed(a=None)
  total_size = 1
  dim = 0
  for s in shape:
    total_size *= s
    dim += 1
  num_elems = int(non_zero_percentage * float(total_size))
  ra_ids = []
  if non_zero_percentage < 1:
    ra_ids = random.sample(range(0, total_size - 1), num_elems)
  else:
    ra_ids = range(num_elems)
  idx = 0
  ids = [1] * num_elems
  for s in ra_ids:
    ids[idx] = id1D_to_idkD(s, shape)
    idx += 1
  tensor_ind = np.array(ids, dtype=np.int64)

  vals = [random.randint(1, 100 * full_range) / 100 + min_range for e in range(num_elems)]
  tensor_vals = np.array(vals, dtype=np.float32)
  return [tensor_ind, tensor_vals, shape]

