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
"""Tests for Pooling and Pooling Grad. Simplified implementation, only tested for 2x2x2"""
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

class SparseTensorDensePoolingTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, pooling_size, rho_data = 0.1, test_type="Pooling"):
    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    with self.test_session(use_gpu=False) as sess:
      pooling = tf.nn.max_pool3d(d1, pooling_size, pooling_size, "SAME");
      stpooling = sc_module.sparse_tensor_max_pooling(t1ind, t1val, t1sh, pooling_size);
      t1 = time.time()
      expected = sess.run(pooling)
      t2 = time.time()
      sv2 = sess.run(stpooling)
      t3 = time.time()
      expected_grad = sess.run(gen_nn_ops._max_pool3d_grad(d1, expected, expected, pooling_size, pooling_size, "SAME"))
      t4 = time.time()
      stpooling_grad = sc_module.sparse_tensor_max_pooling_grad(t1ind, sv2.out_values, sv2.out_corresponding_indices);
      sv2_grad = sess.run(stpooling_grad)
      t5 = time.time()

      print("test type: ", test_type)
      print("input shape: ", tensor_in_sizes)
      print("time dense: ", t2 - t1)
      print("time sparse: ", t3 - t2)
      print("time dense grad: ", t4 - t3)
      print("time sparse grad: ", t5 - t4)
    value2 = sp.sparse_to_dense(sv2.out_indices, sv2.out_values, sv2.out_shape)
    value2grad = sp.sparse_to_dense(t1ind, sv2_grad, t1sh)
    print("dense result: ", expected)
    print("sparse result: ", value2)
    self.assertArrayNear(expected.flatten(), value2.flatten(), 1e-5)
    print("expected grad: ", expected_grad)
    print("value grad: ", value2grad)
    self.assertArrayNear(expected_grad.flatten(), value2grad.flatten(), 1e-5)

  def testPooling(self):
    self._VerifyValues(
      tensor_in_sizes=[1, 3, 3, 1, 1], #[batch, depth, height, width, in_channels]
      pooling_size = [1, 2, 2, 1, 1],
      rho_data=1)
    '''self._VerifyValues(
      tensor_in_sizes=[1, 11, 12, 10, 1], #[batch, depth, height, width, in_channels]
      pooling_size = [1, 2, 2, 2, 1],
      rho_data=1)
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1], #[batch, depth, height, width, in_channels]
      pooling_size = [1, 2, 2, 2, 1],
      rho_data=0.01)'''

if __name__ == "__main__":
  test.main()
