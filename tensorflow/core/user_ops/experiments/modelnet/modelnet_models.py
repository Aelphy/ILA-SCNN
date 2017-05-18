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
import sparse_layer_definition as ld

def model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = False, train_labels = None, approx = True):
  rho_filter = 1
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 3
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sc1 = ld.create_sparse_conv_relu([3,3,3,1,8], rho_filter, strides, padding, approx, dim, var_list, sparse_data, name = "sc1")
  sc2 = ld.create_sparse_conv_relu([3,3,3,8,14], rho_filter, strides, padding, approx, dim, var_list, sc1, name = "sc2")
  sc3 = ld.create_sparse_conv_relu([3,3,3,14,14], rho_filter, strides, padding, approx, dim, var_list, sc2, name = "sc3")
  sc4 = ld.create_sparse_conv_relu([3,3,3,14,20], rho_filter, strides, padding, approx, dim, var_list, sc3, name = "sc4")
  sc5 = ld.create_sparse_conv_relu([3,3,3,20,20], rho_filter, strides, padding, approx, dim, var_list, sc4, name = "sc5")
  sc6 = ld.create_sparse_conv_relu([3,3,3,20,26], rho_filter, strides, padding, approx, dim, var_list, sc5, name = "sc6")
  sc7 = ld.create_sparse_conv_relu([3,3,3,26,26], rho_filter, strides, padding, approx, dim, var_list, sc6, name = "sc7")
  sc8 = ld.create_sparse_conv_relu([3,3,3,26,32], rho_filter, strides, padding, approx, dim, var_list, sc7, name = "sc8")
  sc9 = ld.create_sparse_conv_relu([3,3,3,32,32], rho_filter, strides, padding, approx, dim, var_list, sc8, name = "sc9")
  sc10 = ld.create_sparse_conv_relu([3,3,3,32,32], rho_filter, strides, padding, approx, dim, var_list, sc9, name = "sc10")
  s_out = sc10
  sd = ld.create_direct_sparse_to_dense(s_out)
  sd_flat = tf.reshape(sd, [batch_size, total_size * 32])
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
  return sd_out
