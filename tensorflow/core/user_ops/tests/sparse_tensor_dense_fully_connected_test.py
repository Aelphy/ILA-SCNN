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
from tensorflow.python.ops import math_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
import tensorflow as tf
import random
import numpy as np
import time
import sparse_tools as sp
from sparse_module import sparse_nn_ops as sc_module

#tf.logging.set_verbosity(tf.logging.INFO)

class SparseTensorDenseFCTest(test.TestCase):

  def _VerifyValues(self, nr_in, nr_weights, rho_data = 0.1, rho_weights = 1):
    tensor_in_sizes = [1] + nr_in
    [data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    s_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
    d_data = sp.sparse_to_dense(data_ind, data_val, data_sh)
    
    weight_sizes = nr_in + nr_weights
    [weight_ind, weight_val, weight_sh] = sp.createRandomSparseTensor(rho_weights, weight_sizes)
    s2 = tf.SparseTensor(indices=weight_ind, values=weight_val, dense_shape=weight_sh)
    d_weight = sp.sparse_to_dense(weight_ind, weight_val, weight_sh)


    [bias_ind, bias_val, bias_sh] = sp.createRandomSparseTensor(rho_weights, nr_weights)
    s3 = tf.SparseTensor(indices=bias_ind, values=bias_val, dense_shape=bias_sh)
    d_bias = sp.sparse_to_dense(bias_ind, bias_val, bias_sh)

    with self.test_session(use_gpu=False) as sess:
      h_d_fc = tf.matmul(d_data, d_weight)
      h_s_fc = math_ops.sparse_matmul(s_data, d_weight)
      t1 = time.time()
      expected = sess.run(h_d_fc)
      t2 = time.time()
      sv2 = sess.run(h_s_fc)
      t3 = time.time()
      print("input shape: ", tensor_in_sizes)
      print("weights shape: ", weight_sizes)
      print("time dense: ", t2 - t1)
      print("time sparse: ", t3 - t2)
      print("sv: ", sv2)
      print("expected: ", expected)
      #print("time dense grad: ", t5 - t4)
      #print("time sparse grad: ", t6 - t5)
    #value2 = sp.sparse_to_dense(sv2.sparse_indices, sv2.sparse_values, sv2.sparse_shape)
    #value2grad = sp.sparse_to_dense(sv2.sparse_indices, sv2_grad, sv2.sparse_shape)
    #print("dense result: ", expected)
    #print("sparse result: ", value2)
    self.assertArrayNear(expected.flatten(), sv2.flatten(), 1e-5)
    #self.assertArrayNear(expected_grad.flatten(), value2grad.flatten(), 1e-5)


  def testFC(self):
    self._VerifyValues(
      nr_in=[100], #[batch, flat shape]
      nr_weights=[10], #[batch, flat shape]
      rho_data=0.01,
      rho_weights=1)

 

if __name__ == "__main__":
  test.main()
