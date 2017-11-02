from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad 
from tensorflow.python.platform import test
import tensorflow as tf
import random
import numpy as np
import time
import sparse_tools as sp
import os
import direct_sparse_grad_ops
from tensorflow.python import debug as tf_debug
from direct_sparse_module import sparse_nn_ops as sc_module

def create_sparse_data_to_direct_sparse(sparse_data, dim):
  sd = sparse_data
  return sc_module.direct_sparse_data_conversion(sd.indices, sd.values, sd.dense_shape, dim)

def create_sparse_filter_to_direct_sparse(sparse_filter, tensor_in_shape, dim, name = ""):
  with tf.variable_scope(name):
    sd = sparse_filter
    return sc_module.direct_sparse_filter_conversion(sd.indices, sd.values, sd.dense_shape, sd.dense_shape, dim=dim)

def create_sparse_conv_layer(sparse_data, filter_in_sizes, strides = 1, padding = "SAME", dim = 5, max_density = 0.5, filter_type = "K-RELU", name = "conv", initializer=None, regularizer=None):
  with tf.variable_scope(name):
    dense_filter_shape = np.prod(filter_in_sizes)
    sd = sparse_data
    dense_filter = tf.ones(filter_in_sizes)
    idx = tf.where(tf.not_equal(dense_filter, 0))
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
    sparse_filter_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_filter, idx), dense_filter.get_shape())
    sf = create_sparse_filter_to_direct_sparse(sparse_filter_tensor, sd.out_shape, dim, name);
    f_ind = tf.get_variable('filter_indices', initializer=sf.out_indices, trainable = False, validate_shape=False)
    f_sh = tf.get_variable('filter_shape', initializer=sf.out_shape, trainable = False, validate_shape=False)
    f_map = tf.get_variable('filter_channel_mapping', initializer=sf.out_channel_mapping, trainable = False, validate_shape=False)
    f_val = tf.get_variable('filter_values', initializer=initializer, regularizer=regularizer, shape=[dense_filter_shape], trainable = True, validate_shape=True)
    return sc_module.direct_sparse_conv_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, f_ind, f_val, f_sh, f_map, strides, padding, dim, max_density, filter_type);

def create_sparse_pooling_layer(sparse_data, pooling_sizes, dim):
  sd = sparse_data
  return sc_module.direct_sparse_max_pooling_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, pooling_sizes, dim);

def create_sparse_unpooling_layer(sparse_data, pooling_data, pooling_sizes, dim):
  sd = sparse_data
  pd = pooling_data
  unpooled_values = sc_module.direct_sparse_unpooling_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, pd.out_indices, pd.out_shape, pd.out_block_channel_mapping, strides, dim)
  return {'out_indices': pd.out_indices, 'out_values': unpooled_values, 'out_shape': pd.out_shape, 'out_block_channel_mapping': pd.out_block_channel_mapping}

def create_direct_sparse_to_dense(sparse_data, dim):
  sd = sparse_data
  return sc_module.direct_sparse_to_dense(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, dim)

def create_direct_dense_to_sparse(dense_data, tensor_in_sizes, dim):
  return sc_module.direct_dense_to_sparse(dense_data, tensor_in_sizes, dim)
