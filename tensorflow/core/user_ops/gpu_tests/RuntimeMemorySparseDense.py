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

def verifyValues(
    tensor_in_sizes,
    filter_in_sizes,
    stride,
    rho_data = 0.1,
    rho_filter = 1,
    padding = 'SAME',
    dim = 5,
    max_density = 0.1,
    num_trials = 3,
    filter_type = 'K-RELU',
    test_type = '',
    dense=True
):
    if isinstance(stride, collections.Iterable):
        strides = [1] + list(stride) + [1]
    else:
        strides = [1, stride, stride, stride, 1]

    out_sizes = np.copy(tensor_in_sizes)
    out_sizes[-1] = filter_in_sizes[-1]
    out_entry_count = np.prod(out_sizes) * max_density
        
    bias = np.zeros([filter_in_sizes[-1]], dtype=np.float32)
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
        approx_scskconv = sc_module.direct_sparse_conv_kd(pd.out_indices, pd.out_values, pd.out_shape, pd.out_block_channel_mapping, pf.out_indices, pf.out_values, pf.out_shape, pf.out_channel_mapping, bias, strides, padding, out_entry_count, dim, max_density, filter_type);
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
    
    time.sleep(1)
    
    if dense:
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

        print("time ratio: ", ts / td)
        return [expected, sv3, ts, td]
        

def do_test(res, f_density, batch_size):
    pid = os.getpid()
    print(pid)

    num_trials = 5
    res = res
    channel_count = 1
    channel_count_out = 8
    filter_res = 3
    batch_size = batch_size
    max_density = 1/res
    in_density = 1/res
    f_density = f_density
    filter_type = 'K-RELU'
    test_type = ''
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



for res in [2**i for i in range(4, 9)]:
    for f_density in [0.1, 0.3, 0.5, 1]:
        for batch in [8]:
            print('========================================================================')
            print('========================================================================')
            print('res = {} f_density = {} batch = {}'.format(res, f_density, batch))
            do_test(res, f_density, batch)

