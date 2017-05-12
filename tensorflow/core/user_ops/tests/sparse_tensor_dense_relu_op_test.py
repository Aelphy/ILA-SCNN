# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Relu and ReluGrad."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
import tensorflow as tf
import random
import numpy as np
import time
import sparse_tools as sp
from sparse_module import sparse_nn_ops as sc_module


class SparseTensorDenseReluTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, rho_data = 0.1, test_type="Relu"):
    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    with self.test_session(use_gpu=False) as sess:
      if test_type == "Relu":
        strelu = sc_module.sparse_relu(t1ind, t1val, t1sh);
        relu = nn_ops.relu(d1)
        t1 = time.time()
        expected = sess.run(relu)
        t2 = time.time()
        sv2 = sess.run(strelu)
        t3 = time.time()
        strelu_grad = sc_module.sparse_relu_grad(t1ind, t1val, sv2.out_indices, sv2.out_values);
        relu_grad = gen_nn_ops._relu_grad(d1, expected)
        t4 = time.time()
        expected_grad = sess.run(relu_grad)
        t5 = time.time()
        sv2_grad = sess.run(strelu_grad)
        t6 = time.time()
      if test_type == "Relu6":
        strelu = sc_module.sparse_relu6(t1ind, t1val, t1sh);
        relu = nn_ops.relu6(d1)
        t1 = time.time()
        expected = sess.run(relu)
        t2 = time.time()
        sv2 = sess.run(strelu)
        t3 = time.time()
        strelu_grad = sc_module.sparse_relu6_grad(t1ind, t1val, sv2.out_indices, sv2.out_values);
        relu_grad = gen_nn_ops._relu6_grad(d1, expected)
        t4 = time.time()
        expected_grad = sess.run(relu_grad)
        t5 = time.time()
        sv2_grad = sess.run(strelu_grad)
        t6 = time.time()
      if test_type == "Elu":
        strelu = sc_module.sparse_elu(t1ind, t1val, t1sh);
        relu = nn_ops.elu(d1)
        t1 = time.time()
        expected = sess.run(relu)
        t2 = time.time()
        sv2 = sess.run(strelu)
        t3 = time.time()
        strelu_grad = sc_module.sparse_elu_grad(t1ind, sv2.out_indices, sv2.out_values, t1val);
        relu_grad = gen_nn_ops._elu_grad(expected, d1)
        t4 = time.time()
        expected_grad = sess.run(relu_grad)
        t5 = time.time()
        sv2_grad = sess.run(strelu_grad)
        t6 = time.time()

      print("test type: ", test_type)
      print("input shape: ", tensor_in_sizes)
      print("time dense: ", t2 - t1)
      print("time sparse: ", t3 - t2)
      print("time dense grad: ", t5 - t4)
      print("time sparse grad: ", t6 - t5)
    value2 = sp.sparse_to_dense(sv2.out_indices, sv2.out_values, sv2.out_shape)
    value2grad = sp.sparse_to_dense(sv2.out_indices, sv2_grad, sv2.out_shape)
    #print("dense result: ", expected)
    #print("sparse result: ", value2)
    self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5)
    self.assertArrayNear(expected_grad.flatten(), value2grad.flatten(), 1e-5)


  def testRelu(self):
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1], #[batch, depth, height, width, in_channels]
      rho_data=0.01,
      test_type='Relu')
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1], #[batch, depth, height, width, in_channels]
      rho_data=0.01,
      test_type='Relu6')
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1], #[batch, depth, height, width, in_channels]
      rho_data=0.01,
      test_type='Elu')

 

if __name__ == "__main__":
  test.main()
