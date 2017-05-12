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
from sparse_module import sparse_nn_ops as sc_module


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
  
    #print("input: \n", d1)
    #print("filter: \n", d2)
    #print("strides: \n", strides)

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    print("input shape", tensor_in_sizes)
    print("filter shape", filter_in_sizes)
    with self.test_session(use_gpu=False) as sess:
      scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, padding, dim, False);
      approx_scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, padding, dim, True);
      conv = nn_ops.conv3d(d1, d2, strides, padding)
      f_conv = nn_ops.conv3d(d1, d2, no_strides, padding)
      t1 = time.time()
      expected = sess.run(conv)
      t2 = time.time()
      print("time dense: ", t2 - t1)
      #nstr_expected = sess.run(f_conv)
      t3 = time.time()
      sv2 = sess.run(scskconv)
      t4 = time.time()
      print("time sparse: ", t4 - t3)
      t5 = time.time()
      sv3 = sess.run(approx_scskconv)
      t6 = time.time()
      print("time approx sparse: ", t6 - t5)

      #print("sparse input: \n", s1.eval())
      
    #print("actual sparse: \n", sv2)
    #print("actual v2 sparse: \n", sv3)
    #print("expected: \n", expected)
    value2 = sp.sparse_to_dense(sv2.out_indices, sv2.out_values, sv2.out_shape)
    value3 = sp.sparse_to_dense(sv3.out_indices, sv3.out_values, sv3.out_shape)
    #print("actual v2 sparse shape: \n", sv2.out_shape)
    #print("actual sparse: \n", value2)
    #print("approx sparse: ", value3)
    #print("expected.shape: \n", expected.shape)
    #print("no stride expected: \n", nstr_expected)
    #self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5) 
    #print("actual 1: \n", value1)

    self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5)
    
    #approximative convolution: compare non-zero entries
    approx_cmp = expected.flatten();
    approx = value3.flatten();
    for i in range(len(approx_cmp)):
      if approx[i] == 0:
        approx_cmp[i] = 0
    self.assertArrayNear(approx_cmp, approx, 1e-5)

  def testConv3D1x1x1Filter(self):
    # These are equivalent to the Conv2D1x1 case.
    
    self._VerifyValues(
      tensor_in_sizes=[1, 3, 4, 1, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 1, 1, 1], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      rho_data=1,
      rho_filter=1,
      padding='VALID')
     
    self._VerifyValues(
      tensor_in_sizes=[1, 7, 12, 9, 2], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 7, 5, 2, 2], #[depth, height, width, in_channels, out_channels] 
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
      tensor_in_sizes=[1, 3, 1, 4, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 1, 3, 1, 1], #[depth, height, width, in_channels, out_channels] 
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
        [output4, output3] = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_grad_v2(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)

        output_dense = output.eval()
        output_sparse = output2.eval()
        output_sparse2 = output3.eval()
        value2 = sp.sparse_to_dense(t2ind, output_sparse, t2sh)
        value3 = sp.sparse_to_dense(t2ind, output_sparse2, t2sh)
      else: # test_type == "INPUT"
        t1 = time.time()        
        output = nn_ops.conv3d_backprop_input_v2(tensor_in_sizes, d2, out_backprop_val, strides, padding)
        t2 = time.time()
        output2 = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
        t3 = time.time()
        [output3, output4] = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_grad_v2(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
        t4 = time.time()
        
        output_dense = output.eval()
        output_sparse = output2.eval()
        output_sparse2 = output3.eval()
        value2 = sp.sparse_to_dense(t1ind, output_sparse, t1sh)
        value3 = sp.sparse_to_dense(t1ind, output_sparse2, t1sh)
       
      #print("input: \n", d1)
      #print("filter: \n", d2)
      #print("grads: \n", d3_)
      #print("output dense: \n", output_dense)
      #print("output sparse: \n", value2)
      #print("output sparse v2: \n", value3)
      

      print("time dense: ", t2 - t1)
      print("time sparse: ", t3 - t2)


    self.assertArrayNear(output_dense.flatten(), value2.flatten(), 1e-5)
    self.assertArrayNear(output_dense.flatten(), value3.flatten(), 1e-5) 

  def testGradientValidPaddingStrideOne(self):   
    self.ConstructAndTestGradient(
      tensor_in_sizes=[2, 14, 15, 8, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 5, 7, 1, 1], #[depth, height, width, in_channels, out_channels] 
      stride=1,
      padding="VALID",
      test_type = "FILTER")
    
    self.ConstructAndTestGradient(
      tensor_in_sizes=[2, 13, 13, 1, 1], #[batch, depth, height, width, in_channels]
      filter_in_sizes=[3, 3, 1, 1, 1], #[depth, height, width, in_channels, out_channels] 
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
