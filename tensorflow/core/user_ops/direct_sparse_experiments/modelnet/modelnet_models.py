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
import direct_sparse_grad_ops
from tensorflow.python import debug as tf_debug
import direct_sparse_layer_definition as ld

def model_modelnet10_8(sparse_data, tensor_in_sizes, train_labels = None, num_classes = 10, scope = "mn8-"):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  sc1 = ld.create_sparse_conv_layer(sd_converted, [3,3,3,1,8], strides, padding, dim, 0.8, "K-ABS", name = scope + "sc1")
  sd = ld.create_direct_sparse_to_dense(sc1, dim)
  sd_flat = tf.reshape(sd, [batch_size, total_size])
  if False:
    conv_out = tf.nn.dropout(sd_flat, 0.5, name="dropout")
  else:
    conv_out = sd_flat
  fc512 = tf.layers.dense(conv_out, 1024)
  fc10 = tf.layers.dense(fc512, num_classes)
  #if train:
  sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=train_labels, name = "softmax_loss"))
  #else:
  p_sd_out = tf.nn.softmax(logits=fc10)
  return [sd_out, p_sd_out]

def model_modelnet10_256(sparse_data, tensor_in_sizes, var_list, train = False, train_labels = None, approx = True):
  '''strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sc1 = ld.create_sparse_conv_relu([3,3,3,1,8], rho_filter, strides, padding, approx, dim, var_list, sparse_data, name = "sc1")
  sp1 = ld.layer_to_sparse_tensor(ld.create_sparse_pooling_layer(sc1, pooling_sizes, dim))
  sc2 = ld.create_sparse_conv_relu([3,3,3,8,16], rho_filter, strides, padding, approx, dim, var_list, sp1, name = "sc2")
  sp2 = ld.layer_to_sparse_tensor(ld.create_sparse_pooling_layer(sc2, pooling_sizes, dim))
  sc3 = ld.create_sparse_conv_relu([3,3,3,16,24], rho_filter, strides, padding, approx, dim, var_list, sp2, name = "sc3")
  sp3 = ld.layer_to_sparse_tensor(ld.create_sparse_pooling_layer(sc3, pooling_sizes, dim))
  sc4 = ld.create_sparse_conv_relu([3,3,3,24,32], rho_filter, strides, padding, approx, dim, var_list, sp3, name = "sc4")
  sp4 = ld.layer_to_sparse_tensor(ld.create_sparse_pooling_layer(sc4, pooling_sizes, dim))
  sc5 = ld.create_sparse_conv_relu([3,3,3,32,40], rho_filter, strides, padding, approx, dim, var_list, sp4, name = "sc5")
  sp5 = ld.layer_to_sparse_tensor(ld.create_sparse_pooling_layer(sc5, pooling_sizes, dim))
  sc6 = ld.layer_to_sparse_tensor(ld.create_sparse_conv_layer([3,3,3,40,48], rho_filter, strides, padding, approx, dim, var_list, sp5, name = "sc6"))
  s_out = sc6
  sd = ld.create_direct_sparse_to_dense(s_out)
  sd_flat = tf.reshape(sd, [batch_size, int(total_size * 48 / 32768)])
  if train:
    conv_out = tf.nn.dropout(sd_flat, 0.5, name="dropout")
  else:
    conv_out = sd_flat
  fc512 = tf.layers.dense(conv_out, 512)
  fc10 = tf.layers.dense(fc512, 10)
  if train:
    sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=train_labels, name = "softmax_loss"))
  else:
    sd_out = tf.nn.softmax(logits=fc10)
  return sd_out'''
  return sparse_data

