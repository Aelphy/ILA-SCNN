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
  
  [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes)
  s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
  d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)

  print("strides: \n", strides)
  print("input shape", tensor_in_sizes)
  print("filter shape", filter_in_sizes)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.7

  with tf.device("/gpu:0"):
    convd = sc_module.direct_sparse_data_conversion(t1ind, t1val, t1sh)
    convf = sc_module.direct_sparse_filter_conversion(t2ind, t2val, t2sh, t1sh)
  with tf.Session(config=config) as sess:
    pd = sess.run(convd)
    pf = sess.run(convf)

  tf.reset_default_graph()

  ts = 0
  with tf.device("/gpu:0"):
    approx_scskconv = sc_module.direct_sparse_conv_kd(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, pf.out_indices, pf.out_values, pf.out_shape, pf.out_channel_mapping, strides, padding, dim, max_density, filter_type);
  with tf.Session(config=config) as sess:
    t6 = time.time()
    sv3 = sess.run(approx_scskconv)
    t5 = time.time()
    for i in range(0, num_trials):
      sess.run(approx_scskconv)
    t6 = time.time()
    ts =  abs(t6 - t5) / max(num_trials,1)
    print("time approx sparse: ", ts)
  tf.reset_default_graph()

  td = 0
  with tf.device("/gpu:0"):
    conv = nn_ops.conv3d(d1, d2, strides, padding)
  with tf.Session(config=config) as sess:
    t22 = time.time()
    expected = sess.run(conv)
    t11 = time.time()
    for i in range(0, num_trials):
      sess.run(conv)
    t22 = time.time()
    td = abs(t22 - t11) / max(num_trials,1)
    print("time dense gpu: ", td)
  tf.reset_default_graph()
  
  print("time ratio: ", ts / td);
 
  [bp_ind, sv3_bp_val, bp_sh] = sp.createRandomSparseTensor(1, [len(sv3.out_values)], 1, 9)
  d3_ = sp.sparse1d_to_dense(sv3.out_indices, sv3_bp_val, sv3.out_shape, sv3.out_block_channel_mapping[-1])
  out_backprop_val = constant_op.constant(d3_)
  
  t_bp1 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = nn_ops.conv3d_backprop_filter_v2(d1, filter_in_sizes,  out_backprop_val, strides, padding)
    res_bp1 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp1 = t_bp1 + t2 - t1
  t_bp1 = t_bp1 / float(num_trials)
  print("time bp1: ", t_bp1)
  
  t_bp2 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = nn_ops.conv3d_backprop_input_v2(tensor_in_sizes, d2, out_backprop_val, strides, padding)
    res_bp2 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp2 = t_bp2 + t2 - t1
  t_bp2 = t_bp2 / float(num_trials)
  print("time bp2: ", t_bp2)
  
  t_bp3 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = sc_module.direct_sparse_conv_kd_backprop(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, pf.out_indices, pf.out_values, pf.out_shape, pf.out_channel_mapping, sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping, sv3_bp_val, strides, padding, dim, max_density)
    res_bp3 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp3 = t_bp3 + t2 - t1
  t_bp3 = t_bp3 / float(num_trials)
  print("time bp3: ", t_bp3)
  print("sparse ratio: ", t_bp3 / (t_bp2 + t_bp1))

  bp_sfg = sp.sparse1d_to_dense(pf.out_indices, res_bp3.filter_grads, pf.out_shape, pf.out_channel_mapping[-1])
  bp_sig = sp.sparse1d_to_dense(pd.out_indices, res_bp3.input_grads, pd.out_shape, pd.out_block_channel_mapping[-1])
  value3 = sp.sparse1d_to_dense(sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping[-1])
  #print("expected", expected)
  #print("sv3", value3)

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
    if approx_cmp[i] > 0 and abs(approx_cmp[i] - approx[i]) > 1e-3:
      if has_error == False:
        first_error = i
      has_error = True
      error_cnt = error_cnt + 1
    elif approx[i] != 0:
      correct_cnt = correct_cnt + 1

  bp_sig_flat = bp_sig.flatten()
  res_bp2_flat = res_bp2.flatten()
  bp_i_error_cnt = 0
  bp_i_correct_cnt = 0
  for i in range(len(bp_sig_flat)):
    if bp_sig_flat[i] != 0:
      if bp_sig_flat[i] == res_bp2_flat[i]:
        bp_i_correct_cnt = bp_i_correct_cnt + 1
      else:
        bp_i_error_cnt = bp_i_error_cnt + 1 

  filter_flat = d2.flatten()
  bp_sfg_flat = bp_sfg.flatten()
  res_bp1_flat = res_bp1.flatten()
  bp_f_error_cnt = 0
  bp_f_correct_cnt = 0
  for i in range(len(filter_flat)):
    if filter_flat[i] != 0:
      if bp_sfg_flat[i] == res_bp1_flat[i]:
        bp_f_correct_cnt = bp_f_correct_cnt + 1
      else:
        bp_f_error_cnt = bp_f_error_cnt + 1 

  
  print("total number of non-zero corrects: ", correct_cnt)
  print("sparse input size: ", len(t1ind))
  print("total number of bpi corrects: ", bp_i_correct_cnt)
  print("sparse filter size: ", len(t2ind))
  print("total number of bpf corrects: ", bp_f_correct_cnt)
  if has_error:
    print("total number of errors: ", error_cnt)
    print("first error: ", first_error)
    return 1
  if bp_i_error_cnt > 0:
    print("total number of  bpi errors: ", bp_i_error_cnt)
  if bp_f_error_cnt > 0:
    print("total number of  bpf errors: ", bp_f_error_cnt)
  print("OK")
  return 0

pid = os.getpid()
print(pid)

num_trials = 3
res = 48
channel_count = 3
channel_count_out = 6
filter_res = 3
batch_size = 1
max_density = 1
in_density = 1/res
f_density = 1
filter_type = "K-RELU"
test_type = ""
ret_value = verifyValues(
  tensor_in_sizes=[batch_size, res, res, 1, channel_count], #[batch, depth, height, width, in_channels]
  filter_in_sizes=[filter_res, filter_res, 1, channel_count, channel_count_out], #[depth, height, width, in_channels, out_channels] 
  stride=1,
  rho_data=1 * in_density,
  rho_filter=1 * f_density,
  padding='SAME',
  max_density=max_density,
  num_trials=num_trials,
  filter_type=filter_type,
  test_type=test_type)

sys.exit(0)
