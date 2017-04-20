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
import os

pid = os.getpid()
print(pid)

sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

#print(dir(sc_module))
print(dir(nn_grad))

raw_input("Press Enter to continue...")

tensor_in_sizes=[1, 100, 100, 100, 1] #[batch, depth, height, width, in_channels]
rho_data = 0.01
test_type='Relu'

[t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)

config = tf.ConfigProto(
			device_count = {'GPU': 0}
	)

with tf.Session(config=config) as sess:
  strelu = sc_module.sparse_relu(t1ind, t1val, t1sh);
  relu = nn_ops.relu(d1)
  t1 = time.time()
  expected = sess.run(relu)
  t2 = time.time()
  sv2 = sess.run(strelu)
  t3 = time.time()

  print("input shape", tensor_in_sizes)
  print("time dense: ", t2 - t1)
  print("time sparse: ", t3 - t2)

#value2 = sp.sparse_to_dense(sv2.sparse_indices, sv2.sparse_values, sv2.sparse_shape)
