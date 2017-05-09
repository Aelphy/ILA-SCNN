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
import sparse_ops
from tensorflow.python import debug as tf_debug
sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

#just a quick test, no nice code



filter_in_sizes=[3, 3, 3, 1, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
rho_data = 0.01
rho_filter=1
padding='SAME'
dim = 3
approx = False
res = 33
batch_size = 5
tensor_in_sizes=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]

def sparse_and_dense_conv_relu_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dense_in, sparse_in_ind, sparse_in_val, approx):  
  [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -1, 1)
  sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
  dense_filter_w = sp.sparse_to_dense(filter1_ind, filter1_weights, filter1_sh)
  dense_filter_weights = tf.constant(dense_filter_w, dtype=tf.float32)
  f_ind = tf.constant(filter1_ind, dtype=tf.int64);
  f_w = tf.constant(filter1_weights, dtype=tf.float32);
  f_sh = tf.constant(filter1_sh, dtype=tf.int64);
  print("tis: ", tensor_in_sizes)
  conv = nn_ops.conv3d(dense_in, dense_filter_weights, strides, padding)
  conv_relu = nn_ops.relu(conv)
  stskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(sparse_in_ind, sparse_in_val, tensor_in_sizes, f_ind, f_w, f_sh, strides, padding, dim, approx);
  stskconv_relu = sc_module.sparse_relu(stskconv.out_indices, stskconv.out_values, stskconv.out_shape);
  return [conv_relu, stskconv_relu];

def sparse_and_dense_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dense_in, sparse_in_ind, sparse_in_val, approx):
  [dc1, sc1] = sparse_and_dense_conv_relu_block(tensor_in_sizes, filter_in_sizes, strides, padding, rho_filter, dim, dense_in, sparse_in_ind, sparse_in_val, approx)
  [dc2, sc2] = sparse_and_dense_conv_relu_block(sc1.out_shape, filter_in_sizes, strides, padding, rho_filter, dim, dc1, sc1.out_indices, sc1.out_values, approx);
  [dc3, sc3] = sparse_and_dense_conv_relu_block(sc2.out_shape, filter_in_sizes, strides, padding, rho_filter, dim, dc2, sc2.out_indices, sc2.out_values, approx);
  pooling_size = [1,2,2,2,1]
  pooling = tf.nn.max_pool3d(dc3, pooling_size, pooling_size, "SAME");
  stpooling = sc_module.sparse_tensor_max_pooling(sc3.out_indices, sc3.out_values, sc3.out_shape, pooling_size);
  return [pooling, stpooling]


if isinstance(stride, collections.Iterable):
	strides = [1] + list(stride) + [1]
else:
	strides = [1, stride, stride, stride, 1]

[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
dense_data = sp.sparse_to_dense(data_ind, data_val, data_sh)


v1_sparseness = sp.checkSparsity(dense_data)
t_in_sh = tf.constant(tensor_in_sizes, dtype=tf.int64);
[d1, s1] = sparse_and_dense_block(t_in_sh, filter_in_sizes, strides, padding, rho_filter, dim, dense_data, data_ind, data_val, approx);
[d2, s2] = sparse_and_dense_block(s1.out_shape, filter_in_sizes, strides, padding, rho_filter, dim, d1, s1.out_indices, s1.out_values, approx);
[d3, s3] = sparse_and_dense_block(s2.out_shape, filter_in_sizes, strides, padding, rho_filter, dim, d2, s2.out_indices, s2.out_values, approx);
[d4, s4] = sparse_and_dense_block(s3.out_shape, filter_in_sizes, strides, padding, rho_filter, dim, d3, s3.out_indices, s3.out_values, approx);
[d5, s5] = sparse_and_dense_block(s4.out_shape, filter_in_sizes, strides, padding, rho_filter, dim, d4, s4.out_indices, s4.out_values, approx);
sd = tf.sparse_to_dense(sparse_indices=s5.out_indices, output_shape=s5.out_shape, sparse_values=s5.out_values, validate_indices=False)
d5_flat = tf.reshape(d5, [batch_size, -1])
sd_flat = tf.reshape(sd, d5_flat.get_shape().as_list())

rand_labels = tf.Variable(tf.random_uniform(d5_flat.get_shape().as_list(), 0, 1, dtype=tf.float32, seed=0))
print("labels: ", rand_labels)

dd_loss = tf.losses.softmax_cross_entropy(rand_labels, d5_flat)
sd_loss = tf.losses.softmax_cross_entropy(rand_labels, sd_flat)

# Create a tensor for training op.
dd_train_op = tf.train.GradientDescentOptimizer(0.01).minimize(dd_loss)

sd_train_op = tf.contrib.layers.optimize_loss(
    sd_loss,
    tf.contrib.framework.get_global_step(),
    optimizer='SGD',
    learning_rate=0.001)


config = tf.ConfigProto(
			device_count = {'GPU': 0},
      inter_op_parallelism_threads=2,
      intra_op_parallelism_threads=1
	)

initall = tf.global_variables_initializer()

with tf.Session(config=config) as sess:
  sess.run(initall)
  t1 = time.time()
  #expected = sess.run(dd_train_op)
  expected = sess.run(dd_loss)
  t2 = time.time()
  t3 = time.time()
  #sv2 = sess.run(sd_train_op)
  sv2 = sess.run(sd_loss)
  t4 = time.time()

#value2 = sp.sparse_to_dense(sv2.out_indices, sv2.out_values, sv2.out_shape)
#v2_sparseness = sp.checkSparsity(value2)

print("input shape", tensor_in_sizes)
print("filter shape", filter_in_sizes)
print("time dense: ", t2 - t1)
print("time sparse: ", t4 - t3)

#print("Density before convolution: ", 1 - v1_sparseness)
#print("Density after convolution: ", 1 - v2_sparseness)
#print("Sparse Output Shape: ", sv2.out_shape)
print("Dense Output Shape: ", d5_flat.get_shape().as_list())

print("values sparse: ", sv2)
print("expected values: ", expected)
