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

from sparse_module import sparse_nn_ops as sc_module
import sparse_ops


def layer_to_sparse_tensor(layer):
  tensor = tf.SparseTensor(indices = layer.out_indices, values = layer.out_values, dense_shape = layer.out_shape)
  return tensor

def create_dense_and_sparse_conv_layer(filter_in_sizes_, rho_filter, strides, padding, approx, dim, var_list, sparse_data, dense_data, name = "conv"):
  filter_in_sizes = np.array(filter_in_sizes_, dtype=np.int64)
  with tf.variable_scope(name):
    [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -5, 10)
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

def create_sparse_conv_layer(filter_in_sizes_, rho_filter, strides, padding, approx, dim, var_list, sparse_data, name = "conv"):
  filter_in_sizes = np.array(filter_in_sizes_, dtype=np.int64)
  with tf.variable_scope(name):
    [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -5, 10)
    sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
    f_ind_var = tf.Variable(filter1_ind, trainable=True, name="filter_indices")
    var_list.append(f_ind_var)
    f_w_var = tf.Variable(filter1_weights, trainable=True, name="filter_weights")
    var_list.append(f_w_var)
    f_sh_var = tf.Variable(filter1_sh, trainable=True, name="filter_shape")
    var_list.append(f_sh_var)
    s5 = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(sparse_data.indices, sparse_data.values, sparse_data.dense_shape, f_ind_var, f_w_var, f_sh_var, strides, padding, dim, approx);
    return s5

def create_sparse_relu_layer(sparse_data, name ="Relu"):
  with tf.variable_scope(name):
    return sc_module.sparse_relu(sparse_data.indices, sparse_data.values, sparse_data.dense_shape);

def create_sparse_pooling_layer(sparse_data, pooling_sizes, dim):
  return sc_module.sparse_tensor_max_pooling(sparse_data.indices, sparse_data.values, sparse_data.dense_shape, pooling_sizes, dim);

def create_dense_relu_layer(dense_data):
  return nn_ops.relu(dense_data);

def create_dense_pooling_layer(dense_data, pooling_sizes, padding = "SAME"):
  return tf.nn.max_pool3d(dense_data, pooling_sizes, pooling_sizes, padding);

def create_direct_sparse_to_dense(sparse_data):
  return  sc_module.direct_sparse_to_dense(sparse_indices=sparse_data.indices, output_shape=sparse_data.dense_shape, sparse_values=sparse_data.values, default_value=0, validate_indices=False)


def modelnet10_8(sparse_data, input_shape, var_list, training_labels, train=True, approx=True):
  padding = "SAME"
  strides = [1,1,1,1,1]
  rho_filter = 1
  num_entries = 1
  for i in range(1,len(input_shape)):
    num_entries = num_entries * input_shape[i]


  sc1_ = create_sparse_conv_layer([3,3,3,1,8], rho_filter, strides, padding, approx, 3, var_list, sparse_data, "sc1")
  sc1 = layer_to_sparse_tensor(sc1_)
  sr1 = layer_to_sparse_tensor(create_sparse_relu_layer(sc1, "sr1"))
  '''sc2 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,8,14], rho_filter, strides, padding, approx, 3, var_list, sr1, "sc2"))
  sr2 = layer_to_sparse_tensor(create_sparse_relu_layer(sc2))
  sc3 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,14,14], rho_filter, strides, padding, approx, 3, var_list, sr2, "sc3"))
  sr3 = layer_to_sparse_tensor(create_sparse_relu_layer(sc3))
  sc4 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,14,20], rho_filter, strides, padding, approx, 3, var_list, sr3, "sc4"))
  sr4 = layer_to_sparse_tensor(create_sparse_relu_layer(sc4))
  sc5 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,20,20], rho_filter, strides, padding, approx, 3, var_list, sr4, "sc5"))
  sr5 = layer_to_sparse_tensor(create_sparse_relu_layer(sc5))
  sc6 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,20,26], rho_filter, strides, padding, approx, 3, var_list, sr5, "sc6"))
  sr6 = layer_to_sparse_tensor(create_sparse_relu_layer(sc6))
  sc7 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,26,26], rho_filter, strides, padding, approx, 3, var_list, sr6, "sc7"))
  sr7 = layer_to_sparse_tensor(create_sparse_relu_layer(sc7)) 
  sc8 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,26,32], rho_filter, strides, padding, approx, 3, var_list, sr7, "sc8"))
  sr8 = layer_to_sparse_tensor(create_sparse_relu_layer(sc8))
  sc9 = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,32,32], rho_filter, strides, padding, approx, 3, var_list, sr8, "sc9"))
  sr9 = layer_to_sparse_tensor(create_sparse_relu_layer(sc9))
  sc10_ = layer_to_sparse_tensor(create_sparse_conv_layer([3,3,3,32,32], rho_filter, strides, padding, approx, 3, var_list, sr9, "sc10"))
  '''
  
  sc10 = sc_module.direct_sparse_to_dense(sparse_indices=s_out.indices, output_shape=s_out.dense_shape, sparse_values=s_out.values, default_value=0, validate_indices=False)
  #sc10 = create_direct_sparse_to_dense(sr1)
  #if train == True:
  dropout = tf.nn.dropout(sc10, 0.5)
  conv_out = tf.reshape(sc10, [input_shape[0], num_entries * 8])
  #else:
  #  conv_out = tf.reshape(sc10, [input_shape[0], num_entries * 32])
  fc512 = tf.layers.dense(inputs=conv_out, units=512, activation=tf.nn.relu)
  fc10 = tf.layers.dense(inputs=fc512, units=10, activation=tf.nn.relu)
  #if train == True:
  softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=training_labels))
  #else:
  #  softmax = tf.nn.softmax(logits = fc10)
  model = softmax
  return model
  
