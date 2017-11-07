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
from math import ceil as ceil
import os
import sys

def verifyValues(tensor_in_sizes, stride, rho_data = 0.1, padding = 'SAME', dim = 5, max_density = 1, num_trials = 3, test_type = ""):
  if isinstance(stride, collections.Iterable):
    strides = [1] + list(stride) + [1]
  else:
    strides = [1, stride, stride, stride, 1]

  no_strides = [1, 1, 1, 1, 1]

  [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
  d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
  
  print("strides: \n", strides)
  print("input shape", tensor_in_sizes)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.7

  #reorder data and generate block index lookup table
  with tf.device("/gpu:0"):
    convd = sc_module.direct_sparse_data_conversion(t1ind, t1val, t1sh)
  with tf.Session(config=config) as sess:
    pd = sess.run(convd)
  tf.reset_default_graph()

  ts = 0
  with tf.device("/gpu:0"):
    approx_scskconv = sc_module.direct_sparse_max_pooling_kd(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, strides, dim);
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
  
  tsu = 0
  with tf.device("/gpu:0"):
    unpooling = sc_module.direct_sparse_unpooling_kd(sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping, pd.out_indices, pd.out_shape, pd.out_block_channel_mapping, strides, dim);
  with tf.Session(config=config) as sess:
    t6 = time.time()
    sv4 = sess.run(unpooling)
    t5 = time.time()
    for i in range(0, num_trials):
      sess.run(unpooling)
    t6 = time.time()
    tsu =  abs(t6 - t5) / max(num_trials,1)
    print("time sparse unpooling: ", tsu)
  tf.reset_default_graph()
  
  td = 0
  with tf.device("/gpu:0"):
    pooling = tf.nn.max_pool3d(d1, strides, strides, "SAME");
  with tf.Session(config=config) as sess:
    t22 = time.time()
    expected = sess.run(pooling)
    t11 = time.time()
    for i in range(0, num_trials):
      sess.run(pooling)
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
      fbp = gen_nn_ops._max_pool3d_grad(d1, expected, out_backprop_val, strides, strides, "SAME")
      #fbp = nn_ops.conv3d_backprop_filter_v2(d1, filter_in_sizes,  out_backprop_val, strides, padding)
    res_bp1 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp1 = t_bp1 + t2 - t1
  t_bp1 = t_bp1 / float(num_trials)
  print("time bp1: ", t_bp1)
  
  t_bp3 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = sc_module.direct_sparse_max_pooling_kd_backprop(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping, sv3_bp_val, strides, dim)
    res_bp3 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp3 = t_bp3 + t2 - t1
  t_bp3 = t_bp3 / float(num_trials)
  print("time bp3: ", t_bp3)

  bp_sig = sp.sparse1d_to_dense(pd.out_indices, res_bp3, pd.out_shape, pd.out_block_channel_mapping[-1])
  #print("dense bp ", res_bp1)
  #print("sparse bp: ", bp_sig)
  
  '''print("sparse bp", bp_sig)
  print("sv3 obcm", sv3.out_block_channel_mapping)
  print("len", len(sv3.out_indices))
  print("pd obcm", pd.out_block_channel_mapping)
  print("len", len(pd.out_indices))
  '''
  t_bp4 = 0
  with tf.Session(config=config) as sess:
    with tf.device("/gpu:0"):
      fbp = sc_module.direct_sparse_unpooling_kd_backprop(sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping, pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, res_bp3, strides, dim)
    res_bp4 = sess.run(fbp)
    for i in range(num_trials):
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
      t_bp4 = t_bp4 + t2 - t1
  t_bp4 = t_bp4 / float(num_trials)
  print("time bp3: ", t_bp4)

  bp_unpool = sp.sparse1d_to_dense(sv3.out_indices, res_bp4, sv3.out_shape, sv3.out_block_channel_mapping[-1])
  #print("bp unpool", bp_unpool)
  
  value3 = sp.sparse1d_to_dense(sv3.out_indices, sv3.out_values, sv3.out_shape, sv3.out_block_channel_mapping[-1])
  #print("result sparse ", value3)
  has_error = False
  approx_cmp = expected.flatten()
  approx = value3.flatten()
  non_zero_count = 0
  for i in range(len(approx_cmp)):
    #if approx[i] == 0:
      #approx_cmp[i] = 0
    #else:
      non_zero_count = non_zero_count + 1
  print("entry count: ", non_zero_count)
  error_cnt = 0
  first_error = 0
  correct_cnt = 0
  for i in range(len(approx_cmp)):
    if abs(approx_cmp[i] - approx[i]) > 1e-3:
      #print("error: %d != %d " % (approx_cmp[i], approx[i]))
      #print("at id ", i)
      if has_error == False:
        first_error = i
      has_error = True
      error_cnt = error_cnt + 1
    elif approx[i] != 0:
      correct_cnt = correct_cnt + 1

  bp_sig_flat = bp_sig.flatten()
  res_bp2_flat = res_bp1.flatten()
  bp_i_error_cnt = 0
  bp_i_correct_cnt = 0
  for i in range(len(approx_cmp)):
    if approx[i] != 0:
      if bp_sig_flat[i] == res_bp2_flat[i]:
        bp_i_correct_cnt = bp_i_correct_cnt + 1
      else:
        bp_i_error_cnt = bp_i_error_cnt + 1
  
  p_flat = pd.out_values.flatten()
  up_flat = sv4.flatten()
  up_i_error_cnt = 0
  up_i_correct_cnt = 0
  for i in range(len(p_flat)):
      if p_flat[i] <= up_flat[i]:
        up_i_correct_cnt = up_i_correct_cnt + 1
      else:
        up_i_error_cnt = up_i_error_cnt + 1
  if dim == 5:
    up_bp_cor = 0
    up_bp_err = 0
    for batch in range(0, tensor_in_sizes[0]):
      for channel in range(0, tensor_in_sizes[4]):
        for x in range(0, int(ceil(float(tensor_in_sizes[1]) / float(strides[1])))):
          for y in range(0, int(ceil(float(tensor_in_sizes[2]) / float(strides[2])))):
            for z in range(0, int(ceil(float(tensor_in_sizes[3]) / float(strides[3])))):
              id_in = (batch, x, y, z, channel)
              inval = value3.item(id_in)
              max_out_val = -100000000000
              for dx in range(0, strides[1]):
                xout = x * strides[1] + dx
                if xout >= d1.shape[1]:
                  continue
                for dy in range(0, strides[2]):
                  yout = y * strides[2] + dy
                  if yout >= d1.shape[2]:
                    continue
                  for dz in range(0, strides[3]):
                    zout = z * strides[3] + dz
                    if zout >= d1.shape[3]:
                      continue
                    id_out = (batch, xout, yout, zout, channel)
                    out_val = d1.item(id_out)
                    max_out_val = max(max_out_val, out_val)
              if max_out_val == -100000000000 or max_out_val == inval:
                up_bp_cor = up_bp_cor + 1
              else:
                up_bp_err = up_bp_err + 1

    print("total number of pooling corrects: ", up_bp_cor)
    print("total number of pooling errors: ", up_bp_err)

  if dim == 5:
    up_bp_cor = 0 
    up_bp_err = 0
    tmp = np.copy(bp_unpool)
    for batch in range(0, tensor_in_sizes[0]):
      for channel in range(0, tensor_in_sizes[4]):
        for x in range(0, int(ceil(float(tensor_in_sizes[1]) / float(strides[1])))):
          for y in range(0, int(ceil(float(tensor_in_sizes[2]) / float(strides[2])))):
            for z in range(0, int(ceil(float(tensor_in_sizes[3]) / float(strides[3])))):
              id_in = (batch, x, y, z, channel)
              inval = bp_unpool.item(id_in)
              sum_out_val = 0 
              for dx in range(0, strides[1]):
                xout = x * strides[1] + dx
                if xout >= bp_sig.shape[1]:
                  continue
                for dy in range(0, strides[2]):
                  yout = y * strides[2] + dy
                  if yout >= bp_sig.shape[2]:
                    continue
                  for dz in range(0, strides[3]):
                    zout = z * strides[3] + dz
                    if zout >= bp_sig.shape[3]:
                      continue
                    id_out = (batch, xout, yout, zout, channel)
                    out_val = bp_sig.item(id_out)
                    sum_out_val = sum_out_val + out_val
              if sum_out_val == inval:
                up_bp_cor = up_bp_cor + 1 
              else:
                up_bp_err = up_bp_err + 1
              tmp[id_in] = sum_out_val
    #print("pbup: ", bp_unpool)
    #print("epbup: ", tmp)
    print("total number of unpooling bp corrects: ", up_bp_cor)
    print("total number of unpooling bp errors: ", up_bp_err)

  print("total number of non-zero corrects: ", correct_cnt)
  print("total number of bpi corrects: ", bp_i_correct_cnt)
  print("total number of unpooling corrects: ", up_i_correct_cnt)
  if has_error:
    print("total number of errors: ", error_cnt)
    print("first error: ", first_error)
  if bp_i_error_cnt > 0:
    print("total number of  bpi errors: ", bp_i_error_cnt)
  if up_i_error_cnt > 0:
    print("total number of  up errors: ", up_i_error_cnt)
    return 1
  print("OK")
  return 0

pid = os.getpid()
print(pid)
#raw_input("Press Enter to continue...")

max_density = 1
num_trials = 1
stride = 2
res = 64
channel_count = 8
batch_size = 1
in_density = 1/res
test_type = ""
ret_value = verifyValues(
  tensor_in_sizes=[batch_size, res, res, res, channel_count], #[batch, depth, height, width, in_channels]
  rho_data=1 * in_density,
  stride=stride,
  padding='SAME',
  max_density=max_density,
  num_trials=num_trials,
  test_type=test_type)

sys.exit(ret_value)
