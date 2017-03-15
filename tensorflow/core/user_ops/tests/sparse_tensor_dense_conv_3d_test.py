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
import time
import sparse_tools as sp

sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

class SparseTensorSparseKernelDenseConv3DTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, dim = 3):
    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    no_strides = [1, 1, 1, 1, 1]

    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(0.01, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    #d1 = tf.sparse_tensor_to_dense(s1)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(1, filter_in_sizes)
    s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
    #d2 = tf.sparse_tensor_to_dense(s2)
    d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)
  

    # print("input: \n", d1)
    # print("filter: \n", d2)
    # print("strides: \n", strides)

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    with self.test_session(use_gpu=False) as sess:
      scconv = sc_module.sparse_tensor_dense_conv3d(t1ind, t1val, t1sh, d2, strides);
      scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, dim);
      conv = nn_ops.conv3d(d1, d2, strides, padding="SAME")
      #f_conv = nn_ops.conv3d(d1, d2, no_strides, padding="SAME")
      t1 = time.time()
      expected = sess.run(conv)
      t2 = time.time()
      #nstr_expected = sess.run(f_conv)
      t3 = time.time()
      sv2 = sess.run(scskconv)
      t4 = time.time()

      print("time dense: ", t2 - t1)
      print("time sparse: ", t4 - t3)


    value2 = sp.sparse_to_dense(sv2.sparse_indices, sv2.sparse_values, sv2.sparse_shape)
    # print("actual v2 sparse: \n", sv2)
    # print("actual v2: \n", value2)
    # print("expected: \n", expected)
    # print("expected.shape: \n", expected.shape)
    # print("no stride expected: \n", nstr_expected)
    self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5) 
    '''
    print("actual 1: \n", value1)
    expected = value.flatten() #TODO remove
    self.assertArrayNear(expected.flatten(), value1.flatten(), 1e-5)
    '''


  def testConv3D1x1x1Filter(self):
    # These are equivalent to the Conv2D1x1 case.
    
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[5, 3, 7, 1, 2], #[depth, height, width, in_channels, out_channels] 
      stride=1)
    

  def ConstructAndTestGradient(self, tensor_in_sizes, filter_in_sizes, stride, dim = 3):
    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]


    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(0.01, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    #d1 = tf.sparse_tensor_to_dense(s1)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(1, filter_in_sizes)
    s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
    #d2 = tf.sparse_tensor_to_dense(s2)
    d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)
    in_shape = tensor_in_sizes
    in_val = constant_op.constant(d1)
    filter_shape = filter_in_sizes
    # Make a convolution op with the current settings, just to easily get
    # the shape of the output.
    with self.test_session(use_gpu=False):
      conv_out = nn_ops.conv3d(d1,
                               d2, strides,
                               padding="SAME")
      out_backprop_shape = conv_out.get_shape().as_list()
      [t3ind, t3val, t3sh] = sp.createRandomSparseTensor(0.5, out_backprop_shape, 0, 1)
      d3 = constant_op.constant(sp.sparse_to_dense(t3ind, t3val, t3sh))
      out_backprop_val = d3
      output = nn_ops.conv3d_backprop_filter_v2(in_val, filter_shape,
                                                out_backprop_val, strides,
                                                padding="SAME")
      err = gradient_checker.compute_gradient_error(
          [in_val, out_backprop_val], [in_shape, out_backprop_shape],
          output, filter_shape)
    print("conv3d_backprop_filter gradient err = %g " % err)
    err_tolerance = 1e-3
    self.assertLess(err, err_tolerance)

  def testInputGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(
      tensor_in_sizes=[1, 5, 6, 7, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 3, 1, 2], #[depth, height, width, in_channels, out_channels] 
      stride=1)



if __name__ == "__main__":
  test.main()
