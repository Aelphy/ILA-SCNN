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
import sparse_ops
from tensorflow.python import debug as tf_debug
from sparse_module import sparse_nn_ops as sc_module

def layer_to_sparse_tensor(layer):
  tensor = tf.SparseTensor(indices = layer.out_indices, values = layer.out_values, dense_shape = layer.out_shape)
  return tensor

def create_dense_and_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, dense_data, name = "conv"):
  with tf.variable_scope(name):
    [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -3, 3)
    sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
    f_ind_var = tf.Variable(filter1_ind, trainable=True, name="filter_indices")
    var_list.append(f_ind_var)
    f_w_var = tf.Variable(filter1_weights, trainable=True, name="filter_weights")
    var_list.append(f_w_var)
    f_sh_var = tf.Variable(filter1_sh, trainable=True, name="filter_shape")
    var_list.append(f_sh_var)
    dense_weights = tf.Variable(sp.sparse_to_dense(filter1_ind, filter1_weights, filter1_sh), trainable=True, name="dense_weights")
    var_list.append(dense_weights)
    sparse_conv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(sparse_data.indices, sparse_data.values, sparse_data.dense_shape, f_ind_var, f_w_var, f_sh_var, strides, padding, dim, approx);
    conv = nn_ops.conv3d(dense_data, dense_weights, strides, padding)
    return [conv, sparse_conv]

def create_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, name = "conv"):
  with tf.variable_scope(name):
    [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -3, 3)
    sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
    f_ind_var = tf.Variable(filter1_ind, trainable=True, name="filter_indices")
    var_list.append(f_ind_var)
    f_w_var = tf.Variable(filter1_weights, trainable=True, name="filter_weights")
    var_list.append(f_w_var)
    f_sh_var = tf.Variable(filter1_sh, trainable=True, name="filter_shape")
    var_list.append(f_sh_var)
    s5 = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(sparse_data.indices, sparse_data.values, sparse_data.dense_shape, f_ind_var, f_w_var, f_sh_var, strides, padding, dim, approx);
    return s5

def create_sparse_relu_layer_(sparse_data):
  res_val = nn_ops.relu(sparse_data.values);
  res = tf.SparseTensor(indices = sparse_data.indices, values = res_val, dense_shape = sparse_data.dense_shape)
  return sc_module.sparse_filter_zero_op(res.indices, res.values, res.dense_shape)

def create_sparse_relu_layer(sparse_data):
  return sc_module.sparse_relu(sparse_data.indices, sparse_data.values, sparse_data.dense_shape);

def create_sparse_pooling_layer(sparse_data, pooling_sizes, dim):
  return sc_module.sparse_tensor_max_pooling(sparse_data.indices, sparse_data.values, sparse_data.dense_shape, pooling_sizes, dim);

def create_dense_relu_layer(dense_data):
  return nn_ops.relu(dense_data);

def create_dense_pooling_layer(dense_data, pooling_sizes, padding = "SAME"):
  return tf.nn.max_pool3d(dense_data, pooling_sizes, pooling_sizes, padding);

def create_sparse_conv_relu(filter_in_sizes_, rho_filter, strides, padding, approx, dim, var_list, sparse_data, name = "conv"):
  filter_in_sizes = np.array(filter_in_sizes_, dtype=np.int64)
  cl_ = create_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, name)
  cl = layer_to_sparse_tensor(cl_)
  return layer_to_sparse_tensor(create_sparse_relu_layer_(cl))

def create_direct_sparse_to_dense(sparse_data):
  return sc_module.direct_sparse_to_dense(sparse_indices=sparse_data.indices, output_shape=sparse_data.dense_shape, sparse_values=sparse_data.values, default_value=0, validate_indices=False)
