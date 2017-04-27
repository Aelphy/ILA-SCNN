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
from tensorflow.python import debug as tf_debug
sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

#just a quick test, no nice code


tensor_in_sizes=[1, 100, 100, 100, 1] #[batch, depth, height, width, in_channels]
filter_in_sizes=[3, 3, 3, 1, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
rho_data = 0.2
rho_filter=1
padding='SAME'
dim = 3


def sparse_and_dense_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dense_in, sparse_in_ind, sparse_in_val):  
  [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes)
  sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
  dense_filter_w = sp.sparse_to_dense(filter1_ind, filter1_weights, filter1_sh)
  dense_filter_weights = tf.constant(dense_filter_w, dtype=tf.float32)
  print("dense filter weights: ", dense_filter_w)
  f_ind = tf.constant(filter1_ind, dtype=tf.int64);
  f_w = tf.constant(filter1_weights, dtype=tf.float32);
  f_sh = tf.constant(filter1_sh, dtype=tf.int64);
  t_in_sh = tf.constant(tensor_in_sizes, dtype=tf.int64);
  conv = nn_ops.conv3d(dense_in, dense_filter_weights, strides, padding)
  scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(sparse_in_ind, sparse_in_val, t_in_sh, f_ind, f_w, f_sh, strides, padding, dim);
  return [conv, scskconv];


if isinstance(stride, collections.Iterable):
	strides = [1] + list(stride) + [1]
else:
	strides = [1, stride, stride, stride, 1]

#p_d_sparse_ind = tf.placeholder(tf.int64)
#p_d_sparse_val = tf.placeholder(tf.float32)
#p_d_dense = tf.placeholder(tf.float32)


[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes, -10, 10)
sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
dense_data = sp.sparse_to_dense(data_ind, data_val, data_sh)

[dc1, sc1] = sparse_and_dense_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dense_data, data_ind, data_val);
[dc2, sc2] = sparse_and_dense_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dc1, sc1.sparse_indices, sc1.sparse_values);
[dc3, sc3] = sparse_and_dense_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dc2, sc2.sparse_indices, sc2.sparse_values);


print("dense data: ", dense_data)


#pid = os.getpid()
#print(pid)

#raw_input("Press Enter to continue...")

config = tf.ConfigProto(
			device_count = {'GPU': 0}
	)

with tf.Session(config=config) as sess:

  t1 = time.time()
  expected = sess.run(dc2)
  t2 = time.time()
  t3 = time.time()
  sv2 = sess.run(sc2)
  t4 = time.time()

print("input shape", tensor_in_sizes)
print("filter shape", filter_in_sizes)
print("time dense: ", t2 - t1)
print("time sparse: ", t4 - t3)

value2 = sp.sparse_to_dense(sv2.sparse_indices, sv2.sparse_values, sv2.sparse_shape)

#print("values sparse: ", value2)
#print("expected values: ", expected)
