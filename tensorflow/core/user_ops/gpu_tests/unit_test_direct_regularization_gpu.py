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

def verifyValues(filter_in_sizes, rho_filter = 1, dim = 5, scale_val = 0.1, bias_val = 0.1, num_trials = 3):
  out_channel_count = filter_in_sizes[-1]
  if isinstance(bias_val, collections.Iterable):
    bias = np.array(bias_val, dtype=np.float32)
  else:
    bias = np.array([bias_val] * out_channel_count, dtype=np.float32)
  scale = np.array(scale_val, dtype=np.float32) 
  [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes)
  s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
  d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)

  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.7

  with tf.device("/gpu:0"):
    convf = sc_module.direct_sparse_filter_conversion(t2ind, t2val, t2sh, t2sh)
  with tf.Session(config=config) as sess:
    pf = sess.run(convf)

  tf.reset_default_graph()

  ts = 0
  with tf.device("/gpu:0"):
    creg = sc_module.direct_sparse_channelwise_biased_l2_regularization(pf.out_indices, pf.out_values, pf.out_shape, pf.out_channel_mapping, scale, bias, dim);
  with tf.Session(config=config) as sess:
    t6 = time.time()
    sv3 = sess.run(creg)
    t5 = time.time()
    for i in range(0, num_trials):
      sess.run(creg)
    t6 = time.time()
    ts =  abs(t6 - t5) / max(num_trials,1)
    print("time approx sparse: ", ts)
  tf.reset_default_graph()

  time.sleep(1)
  reg_loss = 0
  for out_channel in range(out_channel_count):
    reg_loss += np.sum(np.power(d2[:,:,:,:,out_channel] + bias[out_channel], 2)) * scale / 2.

  print(sv3, reg_loss)
  if abs(sv3 - reg_loss) > 0.001:
    print("error")
    return 1
  
  #bp_sfg = sp.sparse1d_to_dense(pf.out_indices, res_bp3.filter_grads, pf.out_shape, pf.out_channel_mapping[-1])

  return 0

pid = os.getpid()
print(pid)

num_trials = 8
channel_count = 1
channel_count_out = 2
filter_res = 3
f_density = 1
scale = 0.1
bias = [0., 0.]
ret_value = verifyValues(
  filter_in_sizes=[filter_res, filter_res, filter_res, channel_count, channel_count_out], #[depth, height, width, in_channels, out_channels] 
  rho_filter=1 * f_density,
  scale_val = scale,
  bias_val = bias,
  num_trials=num_trials)

sys.exit(0)
