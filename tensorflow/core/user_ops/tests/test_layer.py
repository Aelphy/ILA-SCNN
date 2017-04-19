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

raw_input("Press Enter to continue...")

tensor_in_sizes=[1, 100, 100, 100, 1] #[batch, depth, height, width, in_channels]
filter_in_sizes=[3, 3, 3, 1, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
dim = 3
rho_data = 0.01
rho_filter=1
padding='VALID'

dim = 3

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
config = tf.ConfigProto(
			device_count = {'GPU': 0}
	)

with tf.Session(config=config) as sess:


	conv = nn_ops.conv3d(d1, d2, strides, padding)
	f_conv = nn_ops.conv3d(d1, d2, no_strides, padding)
	t1 = time.time()
	expected = sess.run(conv)
	t2 = time.time()
	#nstr_expected = sess.run(f_conv)

with tf.Session(config=config) as sess:
	t3 = time.time()
	scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, padding, dim);
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
