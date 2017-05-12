from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np
import sparse_tools as sp
from sparse_module import sparse_nn_ops as sc_module

rho_data = 0.01
res = 33
batch_size = 5
tensor_in_sizes=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]

[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)

bug_fix_shape = np.array(tensor_in_sizes, dtype=np.int64)
print("bug fix ", bug_fix_shape)
sparse_data = tf.sparse_placeholder(tf.float32, shape=bug_fix_shape)

initall = tf.global_variables_initializer()

config = tf.ConfigProto(
      device_count = {'GPU': 0},
  )


with tf.Session(config=config) as sess:
  sess.run(initall, feed_dict={sparse_data: tf.SparseTensorValue(data_ind, data_val, data_sh)})

with tf.Session() as sess:
  sess.run(initall, feed_dict={sparse_data: random_sparse_data})
