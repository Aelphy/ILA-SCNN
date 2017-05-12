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


filter_in_sizes=[3, 3, 3, 1, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
dim = 3
rho_filter=1
padding='SAME'

num_resolutions = 64
res_step_size = 4
num_trials = 4

all_t_s = [None] * num_resolutions
all_t_as = [None] * num_resolutions
all_t_d = [None] * num_resolutions
all_res = [None] * num_resolutions
all_bp1 = [None] * num_resolutions
all_bp2 = [None] * num_resolutions
all_bp3 = [None] * num_resolutions
all_bp4 = [None] * num_resolutions
all_bp5 = [None] * num_resolutions

if isinstance(stride, collections.Iterable):
  strides = [1] + list(stride) + [1]
else:
  strides = [1, stride, stride, stride, 1]

for res_step in range(1, num_resolutions + 1):
  res = res_step * res_step_size

  tensor_in_sizes=[1, res, res, res, 1] #[batch, depth, height, width, in_channels]
  rho_data = 1. / res

  [t1ind, t1val, t1sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
  s1 = tf.SparseTensor(indices=t1ind, values=t1val, dense_shape=t1sh)
  d1 = sp.sparse_to_dense(t1ind, t1val, t1sh)

  [t2ind, t2val, t2sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes)
  s2 = tf.SparseTensor(indices=t2ind, values=t2val, dense_shape=t2sh)
  d2 = sp.sparse_to_dense(t2ind, t2val, t2sh)

  config = tf.ConfigProto(
        device_count = {'GPU': 0},
#        inter_op_parallelism_threads=2,
#        intra_op_parallelism_threads=1
    )

  print("input shape", tensor_in_sizes)
  print("filter shape", filter_in_sizes)
  t_dense = 0
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      conv = nn_ops.conv3d(d1, d2, strides, padding)
      t1 = time.time()
      expected = sess.run(conv)
      t2 = time.time()
    t_dense = t_dense + t2 - t1
  t_dense = t_dense / float(num_trials)

  print("time dense: ", t_dense)

  t_sparse = 0
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, padding, dim, False);
      t1 = time.time()
      sv2 = sess.run(scskconv)
      t2 = time.time()
    t_sparse = t_sparse + t2 - t1
  t_sparse = t_sparse / float(num_trials)
  print("time sparse: ", t_sparse)

  t_asparse = 0
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      scskconv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(t1ind, t1val, t1sh, t2ind, t2val, t2sh, strides, padding, dim, True);
      t1 = time.time()
      sv2 = sess.run(scskconv)
      t2 = time.time()
    t_asparse = t_asparse + t2 - t1
  t_asparse = t_asparse / float(num_trials)
  print("time approx. sparse: ", t_asparse)
  
  t_bp1 = 0
  t_bp2 = 0
  t_bp3 = 0
  t_bp4 = 0
  t_bp5 = 0
  
  out_backprop_shape = conv.get_shape().as_list()
  [t3ind, t3val, t3sh] = sp.createRandomSparseTensor(rho_data + 0.01, out_backprop_shape, 1, 9)
  
  
  if res < 136: #dummy line
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
    print("time bp2: ", t_bp2)
  
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      fbp = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_filter_grad(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
    t_bp3 = t_bp3 + t2 - t1
  t_bp3 = t_bp3 / float(num_trials)
  print("time bp3: ", t_bp3)
  
  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      fbp = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
    t_bp4 = t_bp4 + t2 - t1
  t_bp4 = t_bp4 / float(num_trials)
  print("time bp4: ", t_bp4)

  time.sleep(1)
  with tf.Session(config=config) as sess:
    for i in range(num_trials):
      fbp = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd_grad_v2(t1ind, t1val, t1sh, t2ind, t2val, t2sh, t3ind, t3val, t3sh, strides, padding)
      t1 = time.time()
      sess.run(fbp)
      t2 = time.time()
    t_bp5 = t_bp5 + t2 - t1
  t_bp5 = t_bp5 / float(num_trials)
  
  tf.reset_default_graph()

  print("time bp5: ", t_bp5)
  all_res[res_step - 1] = res
  all_t_s[res_step - 1] = t_sparse
  all_t_as[res_step - 1] = t_asparse
  all_t_d[res_step - 1] = t_dense
  all_bp1[res_step - 1] = t_bp1
  all_bp2[res_step - 1] = t_bp2
  all_bp3[res_step - 1] = t_bp3
  all_bp4[res_step - 1] = t_bp4
  all_bp5[res_step - 1] = t_bp5

result_file = open('eval_time_conv.txt', 'w')
for i in range(len(all_t_s)):
  result_file.write("%s %s %s %s %s %s %s %s %s\n" % (all_res[i], all_t_d[i], all_t_s[i], all_t_as[i], all_bp1[i], all_bp2[i], all_bp3[i], all_bp4[i], all_bp5[i]))

