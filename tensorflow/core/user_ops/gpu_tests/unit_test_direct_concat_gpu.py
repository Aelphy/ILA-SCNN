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

def verifyValues(tensor_in_sizes, rho_data = 0.1, dim = 5, num_trials = 3):
  [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
  d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)
  
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.7

  #reorder data and generate block index lookup table
  with tf.device("/gpu:0"):
    convd = sc_module.direct_sparse_data_conversion(t1ind, t1val, t1sh)
  with tf.Session(config=config) as sess:
    pd = sess.run(convd)
  tf.reset_default_graph()
  values1 = pd.out_values
  [_, values2, _] = sp.createRandomSparseTensor(1, [len(values1)], 1, 3)

  ts = 0
  with tf.device("/gpu:0"):
    concat = sc_module.direct_sparse_concat(pd.out_indices, values1, values2, pd.out_shape, pd.out_block_channel_mapping);
  with tf.Session(config=config) as sess:
    t6 = time.time()
    sv3 = sess.run(concat)
    t5 = time.time()
    for i in range(0, num_trials):
      sess.run(concat)
    t6 = time.time()
    ts =  abs(t6 - t5) / max(num_trials,1)
    print("time sparse concat: ", ts)
  tf.reset_default_graph()
  print(pd)
  #print(values1, values2)
  print(sv3) 
  return 0

pid = os.getpid()
print(pid)
#raw_input("Press Enter to continue...")

num_trials = 1
res = 3
channel_count = 2
batch_size = 2
in_density = 1
ret_value = verifyValues(
  tensor_in_sizes=[batch_size, res, 1, 1, channel_count], #[batch, depth, height, width, in_channels]
  rho_data=1 * in_density,
  num_trials=num_trials)

sys.exit(ret_value)
