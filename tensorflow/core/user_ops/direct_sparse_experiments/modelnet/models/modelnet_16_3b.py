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

def placeholder_inputs(batch_size, num_classes, tensor_in_sizes_):
  tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)  
  batch_label_sizes = [batch_size, num_classes]
  pointclouds_pl = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")
  labels_pl = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
  return pointclouds_pl, labels_pl

def get_model(sparse_data, train_labels, is_training, tensor_in_sizes, num_classes = 10, scope = "mn16-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  ops = [None]*6
  d1 = 0.125
  net = ld.create_sparse_conv_layer(sd_converted, [5,5,5,1,8], tensor_in_sizes, strides, padding, dim, d1, "K-RELU", name = scope + "sc1", initializer=initializer)
  net = ld.create_sparse_conv_layer(net, [5,5,5,8,8], tensor_in_sizes, strides, padding, dim, d1, "K-RELU", name = scope + "sc2", initializer=initializer)
  net = ld.create_sparse_conv_layer(net, [5,5,5,8,8], tensor_in_sizes, strides, padding, dim, d1, "K-RELU", name = scope + "sc3", initializer=initializer)
  [net, tensor_in_sizes] = ld.create_sparse_pooling_layer(net, pooling_sizes, tensor_in_sizes, dim, 6 * d1)
  net = ld.create_sparse_conv_layer(net, [5,5,5,8,16], tensor_in_sizes, strides, padding, dim, 2 * d1, "K-RELU", name = scope + "sc4", initializer=initializer)
  net = ld.create_sparse_conv_layer(net, [5,5,5,16,16], tensor_in_sizes, strides, padding, dim, 2 * d1, "K-RELU", name = scope + "sc5", initializer=initializer)
  net = ld.create_sparse_conv_layer(net, [5,5,5,16,16], tensor_in_sizes, strides, padding, dim, 2 * d1, "K-ABS", name = scope + "sc6", initializer=initializer)
  sd = ld.create_direct_sparse_to_dense(net, dim)
  sd_flat = tf.reshape(sd, [batch_size, total_size * 2])
  conv_out =  tf.layers.dropout(sd_flat, 0.5, name="dropout", training=is_training)
  fc512 = tf.layers.dense(conv_out, 1024, name="dense2")
  fc10 = tf.layers.dense(fc512, num_classes, name="dense1")
  #if train:
  sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=train_labels, name = "softmax_loss"))
  p_sd_out = tf.nn.softmax(logits=fc10)
  return [sd_out, p_sd_out, ops]
