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
from sparse_module import sparse_nn_ops as sc_module

pid = os.getpid()
print(pid)

#print(dir(sc_module))

raw_input("Press Enter to continue...")


pooling_size=[1, 2, 2, 2, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
dim = 3
rho_filter=1
padding='SAME'

num_resolutions = 64
res_step_size = 4
num_trials = 4

all_t_s = [None] * num_resolutions
all_t_d = [None] * num_resolutions
all_res = [None] * num_resolutions
all_bp1 = [None] * num_resolutions
all_bp2 = [None] * num_resolutions


for res_step in range(3, num_resolutions + 1):
  res = res_step * res_step_size

  tensor_in_sizes=[1, res, res, res, 1] #[batch, depth, height, width, in_channels]
  rho_data = 1. / res

  [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
  s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
  d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)

  config = tf.ConfigProto(
        device_count = {'GPU': 0},
#        inter_op_parallelism_threads=2,
#        intra_op_parallelism_threads=1
    )

  print("input shape", tensor_in_sizes)
  print("pooling shape", pooling_size)
  t_dense = 0
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      conv = tf.nn.max_pool3d(d1, pooling_size, pooling_size, "SAME");
      t1 = time.time()
      expected = sess.run(conv)
      t2 = time.time()
    t_dense = t_dense + t2 - t1
  t_dense = t_dense / float(num_trials)
  print("time dense: ", t_dense)

#  print("expected: ", expected)

  t_sparse = 0
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      scskconv = sc_module.sparse_tensor_max_pooling(t1ind, t1val, t1sh, pooling_size);
      t1 = time.time()
      sv2 = sess.run(scskconv)
      t2 = time.time()
    t_sparse = t_sparse + t2 - t1
  t_sparse = t_sparse / float(num_trials)
  print("time sparse: ", t_sparse)
  
#  value2 = sp.sparse_to_dense(sv2.out_indices, sv2.out_values, sv2.out_shape)
#  print("out: ", value2)

  t_bp1 = 0
  t_bp2 = 0
  
  '''[t3ind, t3val, t3sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes, 1, 9)
  
  
  d3_ = sp.sparse_to_dense(t3ind, t3val, t3sh)
  d3 = constant_op.constant(d3_)
  out_backprop_val = d3
  
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      fbp = nn_ops.conv3d_backprop_filter_v2(d1, filter_in_sizes,  out_backprop_val, strides, padding)
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
    t_bp1 = t_bp1 + t2 - t1
  t_bp1 = t_bp1 / float(num_trials)
  print("time bp1: ", t_bp1)

  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      fbp = nn_ops.conv3d_backprop_input_v2(tensor_in_sizes, d2, out_backprop_val, strides, padding)
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
    t_bp2 = t_bp2 + t2 - t1
  t_bp2 = t_bp2 / float(num_trials)
  print("time bp2: ", t_bp2)'''
  
  tf.reset_default_graph()

  all_res[res_step - 1] = res
  all_t_s[res_step - 1] = t_sparse
  all_t_d[res_step - 1] = t_dense
  all_bp1[res_step - 1] = t_bp1
  all_bp2[res_step - 1] = t_bp2

result_file = open('eval_time_pooling.txt', 'w')
for i in range(len(all_t_s)):
  result_file.write("%s %s %s %s %s\n" % (all_res[i], all_t_d[i], all_t_s[i], all_bp1[i], all_bp2[i]))

