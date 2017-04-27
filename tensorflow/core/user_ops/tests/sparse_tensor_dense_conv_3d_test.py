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

tf.logging.set_verbosity(tf.logging.INFO)

sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

class SparseTensorSparseKernelDenseConv3DTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, rho_data = 0.1, rho_filter = 1, padding = 'SAME', dim = 3):
    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    no_strides = [1, 1, 1, 1, 1]

    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes)
    s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
    d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)
  
    # print("input: \n", d1)
    # print("filter: \n", d2)
    # print("strides: \n", strides)

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    with self.test_session(use_gpu=False) as sess:
      scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, padding, dim);
      conv = nn_ops.conv3d(d1, d2, strides, padding)
      f_conv = nn_ops.conv3d(d1, d2, no_strides, padding)
      t1 = time.time()
      expected = sess.run(conv)
      t2 = time.time()
      #nstr_expected = sess.run(f_conv)
      t3 = time.time()
      sv2 = sess.run(scskconv)
      t4 = time.time()
    
      print("input shape", tensor_in_sizes)
      print("filter shape", filter_in_sizes)
      print("time dense: ", t2 - t1)
      print("time sparse: ", t4 - t3)

    value2 = sp.sparse_to_dense(sv2.sparse_indices, sv2.sparse_values, sv2.sparse_shape)
    #print("actual v2 sparse: \n", sv2)
    #print("actual v2 sparse shape: \n", sv2.sparse_shape)
    #print("actual v2: \n", value2)
    #print("expected: \n", expected)
    #print("expected.shape: \n", expected.shape)
    #print("no stride expected: \n", nstr_expected)
    #self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5) 
    #print("actual 1: \n", value1)

    self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5)


  def testConv3D1x1x1Filter(self):
    # These are equivalent to the Conv2D1x1 case.        
    self._VerifyValues(
      tensor_in_sizes=[1, 7, 8, 9, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 4, 1, 1], #[depth, height, width, in_channels, out_channels] 
      stride=2,
      rho_data=1,
      rho_filter=1,
      padding='VALID')
    
    self._VerifyValues(
      tensor_in_sizes=[1, 4, 5, 4, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 4, 1, 1], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      rho_data=1,
      rho_filter=1,
      padding='SAME')
    
    self._VerifyValues(
      tensor_in_sizes=[1, 5, 4, 7, 2], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 3, 2, 2], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      rho_data=1,
      rho_filter=1,
      padding='VALID')
    
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 3, 1, 1], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      rho_data=0.01,
      rho_filter=1,
      padding='VALID')

  def ConstructAndTestGradient(self, tensor_in_sizes, filter_in_sizes, stride, padding = "SAME", test_type = "FILTER", dim = 3):
    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]


    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(1, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(1, filter_in_sizes)
    s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
    d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)
    in_shape = tensor_in_sizes
    in_val = constant_op.constant(d1)
    filter_shape = filter_in_sizes
    # Make a convolution op with the current settings, just to easily get
    # the shape of the output.
    with self.test_session(use_gpu=False):
      conv_out = nn_ops.conv3d(d1, d2, strides, padding)
      out_backprop_shape = conv_out.get_shape().as_list()
      [t3ind, t3val, t3sh] = sp.createRandomSparseTensor(1, out_backprop_shape, 1, 9)
      d3_ = sp.sparse_to_dense(t3ind, t3val, t3sh)
      d3 = constant_op.constant(d3_)
      out_backprop_val = d3

      if test_type == "FILTER":
        t1 = time.time()
        output = nn_ops.conv3d_backprop_filter_v2(in_val, filter_shape,
                                                  out_backprop_val, strides, padding)
        t2 = time.time()
        output2 = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_filter_grad(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
        t3 = time.time()

        output_dense = output.eval()
        output_sparse = output2.eval()
        value2 = sp.sparse_to_dense(t2ind, output_sparse, t2sh)
      else: # test_type == "INPUT"
        t1 = time.time()        
        output = nn_ops.conv3d_backprop_input_v2(tensor_in_sizes, d2, out_backprop_val, strides, padding)
        t2 = time.time()
        output2 = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
        t3 = time.time()
        
        output_dense = output.eval()
        output_sparse = output2.eval()
        value2 = sp.sparse_to_dense(t1ind, output_sparse, t1sh)
       
      '''print("input: \n", d1)
      print("filter: \n", d2)
      print("grads: \n", d3_)
      print("output dense: \n", output_dense)
      print("output sparse: \n", value2)
      '''

      print("time dense: ", t2 - t1)
      print("time sparse: ", t3 - t2)


    self.assertArrayNear(output_dense.flatten(), value2.flatten(), 1e-5) 

  def testGradientValidPaddingStrideOne(self):   
    self.ConstructAndTestGradient(
      tensor_in_sizes=[1, 13, 11, 12, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 4, 5, 1, 2], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      padding="VALID",
      test_type = "FILTER")
    
    self.ConstructAndTestGradient(
      tensor_in_sizes=[2, 14, 13, 12, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 3, 1, 1], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      padding="SAME",
      test_type = "FILTER")
    

    self.ConstructAndTestGradient(
      tensor_in_sizes=[2, 5, 5, 6, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 5, 1, 2], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      padding="SAME",
      test_type = "INPUT")

if __name__ == "__main__":
  test.main()
