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
import direct_sparse_cwise_regularizers as creg
import tensorflow as tf
import random
import numpy as np
import time
import sparse_tools as sp
import os
import direct_sparse_grad_ops
from tensorflow.python import debug as tf_debug
from direct_sparse_module import sparse_nn_ops as sc_module

class DirectSparseData:
  def __init__(self, indices, values, shape, mapping):
    self.out_indices = indices
    self.out_values = values
    self.out_shape = shape
    self.out_block_channel_mapping = mapping

def create_sparse_data_to_direct_sparse(sd, dim):
  return sc_module.direct_sparse_data_conversion(sd.indices, sd.values, sd.dense_shape, dim)

def create_sparse_filter_to_direct_sparse(sparse_filter, tensor_in_shape, dim, name = ""):
  with tf.variable_scope(name):
    sd = sparse_filter
    return sc_module.direct_sparse_filter_conversion(sd.indices, sd.values, sd.dense_shape, sd.dense_shape, dim=dim)

def create_sparse_conv_layer(sparse_data, filter_in_sizes, tensor_in_sizes_, strides = 1, padding = "SAME", dim = 5, max_density = 0.5, filter_type = "K-RELU", name = "conv", initializer=None):
  with tf.variable_scope(name):
    #2. define initialization of sparse filter weights
    tensor_in_sizes = tensor_in_sizes_.copy()
    tensor_in_sizes[-1] = filter_in_sizes[-1]
    dense_filter_shape = np.prod(filter_in_sizes)
    sd = sparse_data
    dense_filter = tf.ones(filter_in_sizes)
    idx = tf.where(tf.not_equal(dense_filter, 0))
    sparse_filter_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_filter, idx), dense_filter.get_shape())
    sf = create_sparse_filter_to_direct_sparse(sparse_filter_tensor, sd.out_shape, dim, name);
    f_ind = tf.get_variable('filter_indices', initializer=sf.out_indices, trainable=False, validate_shape=False)
    f_sh = tf.get_variable('filter_shape', initializer=sf.out_shape, trainable=False, validate_shape=False)
    f_map = tf.get_variable('filter_channel_mapping', initializer=sf.out_channel_mapping, trainable=False, validate_shape=False)
    f_val = tf.get_variable('filter_values', initializer=initializer, shape=[dense_filter_shape], trainable=True, validate_shape=True)
    out_channel_count = filter_in_sizes[-1]
    bias = tf.get_variable('sparse_bias', initializer=tf.zeros_initializer(), shape=[out_channel_count], trainable=True, validate_shape=True) #TODO: make trainable
    #3. define convolutional layer
    sparse_entry_bound = np.prod(tensor_in_sizes) * max_density + filter_in_sizes[-1]
    conv_layer = sc_module.direct_sparse_conv_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, f_ind, f_val, f_sh, f_map, bias, strides, padding, sparse_entry_bound, dim, max_density, filter_type)
    return conv_layer, tensor_in_sizes

def create_sparse_conv_layer_reg(sparse_data, filter_in_sizes, tensor_in_sizes_, strides = 1, padding = "SAME", dim = 5, max_density = 0.5, filter_type = "K-RELU", name = "conv", initializer=None, scale=0.005, bias_offset=0.005, bias_momentum = 0.95):
  with tf.variable_scope(name):
    out_channel_count = filter_in_sizes[-1]
    tensor_in_sizes = tensor_in_sizes_.copy()
    tensor_in_sizes[-1] = filter_in_sizes[-1]
    max_de = tf.constant(max_density, dtype=tf.float32)
    min_bias = tf.constant(-bias_offset, dtype=tf.float32)
    max_bias = tf.constant(bias_offset, dtype=tf.float32)
    bias_offset = tf.constant(bias_offset, dtype=tf.float32)
    reg_bias = tf.get_variable('regularisation_bias', initializer=tf.zeros_initializer(), shape=[out_channel_count], trainable = False, dtype=tf.float32)

    #2. define initialization of sparse filter weights
    dense_filter_shape = np.prod(filter_in_sizes)
    sd = sparse_data
    dense_filter = tf.ones(filter_in_sizes)
    idx = tf.where(tf.not_equal(dense_filter, 0))
    sparse_filter_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_filter, idx), dense_filter.get_shape())
    sf = create_sparse_filter_to_direct_sparse(sparse_filter_tensor, sd.out_shape, dim, name);
    f_ind = tf.get_variable('filter_indices', initializer=sf.out_indices, trainable=False, validate_shape=False)
    f_sh = tf.get_variable('filter_shape', initializer=sf.out_shape, trainable=False, validate_shape=False)
    f_map = tf.get_variable('filter_channel_mapping', initializer=sf.out_channel_mapping, trainable=False, validate_shape=False)
    regularizer = creg.biased_l2_regularizer(scale, reg_bias, f_ind, f_sh, f_map)
    f_val = tf.get_variable('filter_values', initializer=initializer, regularizer=regularizer, shape=[dense_filter_shape], trainable=True, validate_shape=True)
    bias = tf.get_variable('sparse_bias', initializer=tf.zeros_initializer(), shape=[out_channel_count], trainable=True, validate_shape=True) #TODO: make trainable
    #3. define convolutional layer
    sparse_entry_bound = np.prod(tensor_in_sizes) * max_density + filter_in_sizes[-1]
    conv_layer = sc_module.direct_sparse_conv_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, f_ind, f_val, f_sh, f_map, bias, strides, padding, sparse_entry_bound, dim, max_density, filter_type)

    out_density = conv_layer.out_channel_densities # TODO: custom regularizer per channel
    density_ge = tf.greater_equal(out_density, max_de)
    new_bias = tf.cast(tf.where(density_ge, max_bias * (out_density - max_de) + bias_offset, min_bias * (max_de - out_density)), dtype=tf.float32)
    new_bias = (1 - bias_momentum) * new_bias + bias_momentum * reg_bias; #lowpass filter
    assign_op = tf.assign(reg_bias, new_bias, validate_shape=False, name='update_bias')
    return conv_layer, tensor_in_sizes, assign_op

def create_sparse_pooling_layer(sparse_data, pooling_sizes, tensor_in_sizes, dim, max_density = 0.):
  sd = sparse_data
  pooling = sc_module.direct_sparse_max_pooling_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, pooling_sizes, max_density, dim);
  tensor_in_sizes = np.divide(tensor_in_sizes, pooling_sizes)
  return [pooling, tensor_in_sizes]

def create_sparse_unpooling_layer(sparse_data, pooling_data, pooling_sizes, dim):
  sd = sparse_data
  pd = pooling_data
  unpooled_values = sc_module.direct_sparse_unpooling_kd(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, pd.out_indices, pd.out_shape, pd.out_block_channel_mapping, strides, dim)
  return DirectSparseData(pd.out_indices, unpooled_values, pd.out_shape, pd.out_block_channel_mapping)

def create_direct_sparse_to_dense(sparse_data, dim):
  sd = sparse_data
  return sc_module.direct_sparse_to_dense(sd.out_indices, sd.out_values, sd.out_shape, sd.out_block_channel_mapping, dim)

def create_direct_dense_to_sparse(dense_data, tensor_in_sizes, dim):
  return sc_module.direct_dense_to_sparse(dense_data, tensor_in_sizes, dim)

def create_direct_sparse_batchnorm(sparse_data, name='batch_norm'):
  with tf.variable_scope(name):
    sd = sparse_data
    #TODO: create sparse batch norm
    batch_norm = tf.layers.batch_normalization(sparse_data.out_values)
    return DirectSparseData(sd.out_indices, batch_norm, sd.out_shape, sd.out_block_channel_mapping)
