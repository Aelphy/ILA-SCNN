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

def get_model(sparse_data, train_labels, is_training, tensor_in_sizes, num_classes = 10, scope = "mn128-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5 
  pooling_sizes = [1,2,2,2,1]
  dpooling_sizes = [2,2,2]
  batch_size = tensor_in_sizes[0]
  total_size = 1 
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  ops = [None]*9
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  d1 = 0.02
  net, tensor_in_sizes = ld.create_sparse_conv_layer(sd_converted, [3,3,3,1,8], tensor_in_sizes, strides, padding, dim, d1, "K-RELU", name = scope + "sc1", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,8,8], tensor_in_sizes, strides, padding, dim, d1, "K-RELU", name = scope + "sc2", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,8,8], tensor_in_sizes, strides, padding, dim, d1, "K-RELU", name = scope + "sc3", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_pooling_layer(net, pooling_sizes, tensor_in_sizes, dim, 0.06)
  d2 = 0.06
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,8,16], tensor_in_sizes, strides, padding, dim, d2, "K-RELU", name = scope + "sc4", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,16,16], tensor_in_sizes, strides, padding, dim, d2, "K-RELU", name = scope + "sc5", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,16,16], tensor_in_sizes, strides, padding, dim, d2, "K-RELU", name = scope + "sc6", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_pooling_layer(net, pooling_sizes, tensor_in_sizes, dim, 0.18)
  d3 = 0.14
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,16,24], tensor_in_sizes, strides, padding, dim, d3, "K-RELU", name = scope + "sc7", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,24,24], tensor_in_sizes, strides, padding, dim, d3, "K-RELU", name = scope + "sc8", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_conv_layer(net, [3,3,3,24,24], tensor_in_sizes, strides, padding, dim, d3, "K-RELU", name = scope + "sc9", initializer=initializer)
  net, tensor_in_sizes = ld.create_sparse_pooling_layer(net, pooling_sizes, tensor_in_sizes, dim, 0.50)
  net = ld.create_direct_sparse_to_dense(net, dim)
  net = tf.reshape(net, [batch_size, 16, 16, 16, 24])
  #dense layers
  net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc10", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc11", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc12", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.max_pooling3d(inputs=net, pool_size=dpooling_sizes, strides=2, padding="same", name="dp1")
  net = tf.layers.conv3d(inputs=net, filters=40, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc13", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=40, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc14", kernel_initializer=initializer, kernel_regularizer=regularizer)
  n3 = net = tf.layers.conv3d(inputs=net, filters=40, kernel_size=[3, 3, 3], padding="same", name = scope + "sc15")
  net =  tf.layers.dropout(net, 0.5, name="dropout", training=is_training)
  net = tf.reshape(net, [batch_size, -1])
  net = tf.layers.dense(net, 1024)
  net = tf.layers.dense(net, num_classes)
  sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=train_labels, name = "softmax_loss"))
  p_sd_out = tf.nn.softmax(logits=net)

  
  return [sd_out, p_sd_out, ops]
