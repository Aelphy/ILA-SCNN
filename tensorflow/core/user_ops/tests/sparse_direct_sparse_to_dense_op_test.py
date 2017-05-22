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


class DirectSparseToDenseTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, rho_data = 0.1):
    [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    sd = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
    d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
    
    [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
    d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)

    with self.test_session(use_gpu=False) as sess:
      sparse_to_dense = sc_module.direct_sparse_to_dense(sparse_indices=sd.indices, output_shape=sd.dense_shape, sparse_values=sd.values, default_value=0, validate_indices=False)
      converted = sess.run(sparse_to_dense)
      converted_grad = sess.run(tf.gather_nd(converted, sd.indices))
    #print("converted to dense: ", converted)
    #print("sparse values: ", t1val)
    #print("sparse gradients: ", converted_grad)
    self.assertArrayNear(d1.flatten(), converted.flatten(), 1e-5)
    self.assertArrayNear(t1val.flatten(), converted_grad.flatten(), 1e-5)


  def testSparseToDense(self):
    self._VerifyValues(
      tensor_in_sizes=[1, 3, 2, 1, 1],
      rho_data=1)
    self._VerifyValues(
      tensor_in_sizes=[1, 33, 12, 19, 1],
      rho_data=0.01)
    self._VerifyValues(
      tensor_in_sizes=[1, 100, 100, 100, 1],
      rho_data=0.01)

if __name__ == "__main__":
  test.main()
