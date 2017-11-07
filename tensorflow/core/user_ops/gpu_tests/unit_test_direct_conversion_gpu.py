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
from tensorflow.python.ops import gen_nn_ops
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

def verifyValues(tensor_in_sizes, rho_data = 0.1, dim = 5,  num_trials = 3, test_type = ""):

  [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
  s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
  d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
  
  #print("ind in: \n", t1ind)
  #print("input: \n", d1)

  # Initializes the input tensor with array containing incrementing
  # numbers from 1.
  print("input shape", tensor_in_sizes)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.7

  #reorder data and generate block index lookup table
  td = 0
  with tf.device("/gpu:0"):
    dts = sc_module.direct_dense_to_sparse(d1, tensor_in_sizes, dim);
  with tf.Session(config=config) as sess:
    t22 = time.time()
    pd = sess.run(dts)
    t11 = time.time()
    for i in range(0, num_trials):
      sess.run(dts)
    t22 = time.time()
    td = abs(t22 - t11) / max(num_trials,1)
    print("time dense to sparse gpu: ", td)
  tf.reset_default_graph()
  
  expected = d1
  
  td = 0
  with tf.device("/gpu:0"):
    s2d = sc_module.direct_sparse_to_dense(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping);
  with tf.Session(config=config) as sess:
    t22 = time.time()
    sv3 = sess.run(s2d)
    t11 = time.time()
    for i in range(0, num_trials):
      sess.run(s2d)
    t22 = time.time()
    td = abs(t22 - t11) / max(num_trials,1)
    print("time sparse to dense gpu: ", td)
  tf.reset_default_graph()
  
 
  [bp_ind, sv3_bp_val, bp_sh] = sp.createRandomSparseTensor(1, tensor_in_sizes, 1, 9)
  d3_ = sp.sparse1d_to_dense(pd.out_indices, sv3_bp_val, pd.out_shape, pd.out_block_channel_mapping[-1])
  out_backprop_val = constant_op.constant(d3_)
  
  t_bp3 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = sc_module.direct_sparse_to_dense_backprop(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, sv3, out_backprop_val)
    res_bp3 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp3 = t_bp3 + t2 - t1
  t_bp3 = t_bp3 / float(num_trials)
  print("time bp sparse to dense: ", t_bp3)
  
  t_bp4 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = sc_module.direct_dense_to_sparse_backprop(sv3, pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, res_bp3)
    res_bp4 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp4 = t_bp3 + t2 - t1
  t_bp4 = t_bp4 / float(num_trials)
  print("time bp dense to sparse: ", t_bp4)

  bp_sig = sp.sparse1d_to_dense(pd.out_indices, res_bp3, pd.out_shape, pd.out_block_channel_mapping[-1])
  #print("dense bp ", res_bp1)
  #print("sparse bp: ", bp_sig)
  
  has_error = False
  approx_cmp = expected.flatten()
  approx = sv3.flatten()
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
  ebp = d3_.flatten()
  #rbp = bp_sig.flatten()
  rbp = res_bp4.flatten()
  bperror_cnt = 0
  bpcorrect_cnt = 0
  for i in range(len(ebp)):
    if abs(ebp[i] - rbp[i]) > 1e-3:
      bperror_cnt = bperror_cnt + 1
    elif rbp[i] != 0:
      bpcorrect_cnt = bpcorrect_cnt + 1

  print("total number of non-zero corrects: ", correct_cnt)
  print("total number of backprop corrects: ", bpcorrect_cnt)
  if has_error:
    print("total number of errors: ", error_cnt)
    print("first error: ", first_error)
    return 1
  if bperror_cnt > 0:
    print("total number of backprop errors: ", bperror_cnt)
  print("OK")
  return 0


pid = os.getpid()
print(pid)
#raw_input("Press Enter to continue...")

num_trials = 1
res = 50
channel_count = 2
batch_size = 2
in_density = 1 / res
test_type = ""
ret_value = verifyValues(
  tensor_in_sizes=[batch_size, res, res, res, channel_count], #[batch, depth, height, width, in_channels]
  rho_data=1 * in_density,
  num_trials=num_trials,
  test_type=test_type)

sys.exit(0)
