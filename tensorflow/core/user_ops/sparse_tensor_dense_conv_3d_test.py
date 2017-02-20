from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
import tensorflow as tf
import random
import numpy as np

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

def createRandomSparseTensor(non_zero_percentage, shape):
  random.seed(a=None)
  total_size = 1
  dim = 0
  for s in shape:
    total_size *= s
    dim += 1
  num_elems = int(non_zero_percentage * float(total_size))
  ra_ids = random.sample(range(0, total_size - 1), num_elems)
  idx = 0
  ids = [1] * num_elems
  for s in ra_ids:
    ids[idx] = id1D_to_idkD(s, shape)
    idx += 1
  tensor_ind = np.array(ids, dtype=np.int64)

  vals = [1] * num_elems
  #vals = random.sample(range(1, 255), num_elems)
  tensor_vals = np.array(vals, dtype=np.float32)
  return [tensor_ind, tensor_vals, shape]


class SparseTensorSparseKernelDenseConv3DTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride):
    sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    [t1ind, t1val, t1sh] = createRandomSparseTensor(0.2, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    #d1 = tf.sparse_tensor_to_dense(s1)
    d1 = sparse_to_dense(t1ind, t1val, t1sh)
    
    [t2ind, t2val, t2sh] = createRandomSparseTensor(0.4, filter_in_sizes)
    s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
    #d2 = tf.sparse_tensor_to_dense(s2)
    d2 = sparse_to_dense(t2ind, t2val, t2sh)
  

    #print("t1ind: \n", t1ind)
    #print("t1val \n", t1val)
    #print("t1sh: \n", t1sh)
    print("input: \n", d1)
    #print("t2ind: \n", t2ind)
    #print("t2val \n", t2val)
    #print("t2sh: \n", t2sh)
    print("filter: \n", d2)
    print("strides: \n", strides)

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    with self.test_session(use_gpu=True) as sess:
      scconv = sc_module.sparse_tensor_dense_conv3d(t1ind, t1val, t1sh, d2, strides);
      scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv3d(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides);
      conv = nn_ops.conv3d(d1, d2, strides, padding="SAME")
      expected = sess.run(conv)
    #  sv1 = sess.run(scconv)
      sv2 = sess.run(scskconv)
    value2 = sparse_to_dense(sv2.sparse_indices, sv2.sparse_values, sv2.sparse_shape)
    print("actual v2 sparse: \n", sv2)
    print("actual v2: \n", value2)
    print("expected: \n", expected)
    self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5) 
    '''
    print("actual 1: \n", value1)
    expected = value.flatten() #TODO remove
    self.assertArrayNear(expected.flatten(), value1.flatten(), 1e-5)
    '''


  def testConv3D1x1x1Filter(self):
    # These are equivalent to the Conv2D1x1 case.
    '''    
    self._VerifyValues(
      tensor_in_sizes=[1, 1, 3, 3, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[1, 3, 3, 1, 1], #[depth, height, width, output_channels, in_channels] 
      stride=1)
    
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 1, 1],
        filter_in_sizes=[1, 2, 3, 1, 1],
        stride=1)
    '''
    self._VerifyValues(
        tensor_in_sizes=[1, 1, 1, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1)
    
  # Expected values computed using scipy's correlate function.
'''  def testConv3D2x2x2Filter(self):
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
        filter_in_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
        stride=1)

  def testConv3DStrides(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 5, 8, 7, 1],
        filter_in_sizes=[1, 2, 3, 1, 1],
        stride=[2, 3, 1])

  def testConv3D2x2x2FilterStride2(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2)

  def testConv3DStride3(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 6, 7, 8, 2],
        filter_in_sizes=[3, 2, 1, 2, 3],
        stride=3)

  def testConv3D2x2x2FilterStride2Same(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2)

  def testKernelSmallerThanStride(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2)
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2)

    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3)

    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3)

  def testKernelSizeMatchesInputSize(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 2, 1],
        filter_in_sizes=[2, 1, 2, 1, 2],
        stride=1)

'''


if __name__ == "__main__":
  test.main()
