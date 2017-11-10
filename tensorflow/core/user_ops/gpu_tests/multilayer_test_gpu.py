#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.client import timeline
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
import tensorflow as tf
import random
import numpy as np
import time
import sparse_tools as sp
from direct_sparse_module import sparse_nn_ops as sc_module
import os
import sys

def verifyValues(tensor_in_sizes, filter_in_sizes, stride, rho_data = 0.1, rho_filter = 1, padding = 'SAME', dim = 5, max_density = 0.1, num_trials = 3, filter_type="K-RELU", test_type = ""):
  if isinstance(stride, collections.Iterable):
    strides = [1] + list(stride) + [1]
  else:
    strides = [1, stride, stride, stride, 1]

  no_strides = [1, 1, 1, 1, 1]
  [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes, -3, 3)
  s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
  d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
  
  [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -3, 3)
  s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
  d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)

  filter_in_sizes2 = filter_in_sizes[:]
  filter_in_sizes2[-2] = filter_in_sizes2[-1]
  [t3ind, t3val, t3sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes2, -3, 3)
  s3 = tf.SparseTensor(indices=t3ind, values=t3val, dense_shape=t3sh)
  d3 = sp.sparse_to_dense(t3ind, t3val, t3sh)

  [t4ind, t4val, t4sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes2, -3, 3)
  s4 = tf.SparseTensor(indices=t4ind, values=t4val, dense_shape=t4sh)
  d4 = sp.sparse_to_dense(t4ind, t4val, t4sh)

  print("strides: \n", strides)
  print("input shape", tensor_in_sizes)
  print("filter shape", filter_in_sizes)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.device("/gpu:0"):
    convd = sc_module.direct_sparse_data_conversion(t1ind, t1val, t1sh)
    convf = sc_module.direct_sparse_filter_conversion(t2ind, t2val, t2sh, t1sh)
    convf2 = sc_module.direct_sparse_filter_conversion(t3ind, t3val, t3sh, t3sh)
    convf3 = sc_module.direct_sparse_filter_conversion(t4ind, t4val, t4sh, t4sh)
  with tf.Session(config=config) as sess:
    pd = sess.run(convd)
    pf = sess.run(convf)
    pf2 = sess.run(convf2)
    pf3 = sess.run(convf3)

  tf.reset_default_graph()

  ts = 0
  with tf.device("/gpu:0"):
    net = sc_module.direct_sparse_conv_kd(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, pf.out_indices, pf.out_values, pf.out_shape, pf.out_channel_mapping, strides, padding, dim, max_density, filter_type);
    net = sc_module.direct_sparse_conv_kd(net.out_indices, net.out_values, net.out_shape, net.out_block_channel_mapping, pf2.out_indices, pf2.out_values, pf2.out_shape, pf2.out_channel_mapping, strides, padding, dim, max_density, filter_type);
    net = sc_module.direct_sparse_conv_kd(net.out_indices, net.out_values, net.out_shape, net.out_block_channel_mapping, pf3.out_indices, pf3.out_values, pf3.out_shape, pf3.out_channel_mapping, strides, padding, dim, max_density, filter_type);
  with tf.Session(config=config) as sess:
    t6 = time.time()
    sv3 = sess.run(net)
    t5 = time.time()
    for i in range(0, num_trials):
      sess.run(net)
    t6 = time.time()
    ts =  abs(t6 - t5) / max(num_trials,1)
    print("time approx sparse: ", ts)
  tf.reset_default_graph()

  td = 0
  with tf.device("/gpu:0"):
    net = nn_ops.conv3d(d1, d2, strides, padding)
    if filter_type == "K-RELU":
      net = nn_ops.relu(net)
    net = nn_ops.conv3d(net, d3, strides, padding)
    if filter_type == "K-RELU":
      net = nn_ops.relu(net)
    net = nn_ops.conv3d(net, d4, strides, padding)
    if filter_type == "K-RELU":
      net = nn_ops.relu(net)
  with tf.Session(config=config) as sess:
    t22 = time.time()
    expected = sess.run(net)
    t11 = time.time()
    for i in range(0, num_trials):
      sess.run(net)
    t22 = time.time()
    td = abs(t22 - t11) / max(num_trials,1)
    print("time dense gpu: ", td)
  tf.reset_default_graph()
  
  value3 = sp.sparse1d_to_dense(sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping[-1])
  print("expected: ", expected)
  print("sparse: ", value3, sv3)
  has_error = False
  approx_cmp = expected.flatten()
  approx = value3.flatten()
  non_zero_count = 0
  for i in range(len(approx_cmp)):
      non_zero_count = non_zero_count + 1
  print("entry count: ", non_zero_count)
  error_cnt = 0
  first_error = 0
  correct_cnt = 0
  for i in range(len(approx_cmp)):
    if abs(approx_cmp[i] - approx[i]) > 1e-3:
      if has_error == False:
        first_error = i
      has_error = True
      error_cnt = error_cnt + 1
    elif approx[i] != 0:
      correct_cnt = correct_cnt + 1

  print("total number of non-zero corrects: ", correct_cnt)
  print("sparse input size: ", len(t1ind))
  if has_error:
    print("total number of errors: ", error_cnt)
    print("first error: ", first_error)
    return 1
  print("OK")
  return 0

pid = os.getpid()
print(pid)

num_trials = 3
res = 10
channel_count = 1
channel_count_out = 8
filter_res = 3
batch_size = 1
max_density = 1
in_density = 1/res
f_density = 1
filter_type = "K-RELU"
test_type = ""
ret_value = verifyValues(
  tensor_in_sizes=[batch_size, res, res, res, channel_count], #[batch, depth, height, width, in_channels]
  filter_in_sizes=[filter_res, filter_res, filter_res, channel_count, channel_count_out], #[depth, height, width, in_channels, out_channels] 
  stride=1,
  rho_data=1 * in_density,
  rho_filter=1 * f_density,
  padding='SAME',
  max_density=max_density,
  num_trials=num_trials,
  filter_type=filter_type,
  test_type=test_type)

sys.exit(0)
