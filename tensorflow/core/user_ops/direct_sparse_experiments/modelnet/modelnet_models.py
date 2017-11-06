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


def model_modelnet_res(res, sparse_data, tensor_in_sizes, train_labels = None, num_classes = 10, scope = "mn", initializer = None, regularizer = None):
  if res == 8:
    return model_modelnet_8(sparse_data, tensor_in_sizes, train_labels, num_classes, scope, initializer, regularizer)
  elif res == 16:
    return model_modelnet_16(sparse_data, tensor_in_sizes, train_labels, num_classes, scope, initializer, regularizer)
  elif res == 256:
    return model_modelnet_256(sparse_data, tensor_in_sizes, train_labels, num_classes, scope, initializer, regularizer)

def model_modelnet_8(sparse_data, tensor_in_sizes, train_labels = None, num_classes = 10, scope = "mn8-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  d1 = 0.25
  net = ld.create_sparse_conv_layer(sd_converted, [3,3,3,1,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc1", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc2", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,8], strides, padding, dim, d1, "K-ABS", name = scope + "sc3", initializer=initializer, regularizer=regularizer)
  sd = ld.create_direct_sparse_to_dense(net, dim)
  sd_flat = tf.reshape(sd, [batch_size, total_size * 8])
  if train_labels != None:
    conv_out = tf.nn.dropout(sd_flat, 0.5, name="dropout")
  else:
    conv_out = sd_flat
  fc512 = tf.layers.dense(conv_out, 1024, name="dense2")
  fc10 = tf.layers.dense(fc512, num_classes, name="dense1")
  #if train:
  if train_labels != None:
    sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=train_labels, name = "softmax_loss"))
  else:
    sd_out = None
  p_sd_out = tf.nn.softmax(logits=fc10)
  return [sd_out, p_sd_out]

def model_modelnet_16(sparse_data, tensor_in_sizes, train_labels = None, num_classes = 10, scope = "mn16-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  d1 = 0.125
  net = ld.create_sparse_conv_layer(sd_converted, [3,3,3,1,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc1", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc2", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc3", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_pooling_layer(net, pooling_sizes, dim, 6 * d1)
  net = ld.create_sparse_conv_layer(sd_converted, [3,3,3,8,16], strides, padding, dim, d1, "K-RELU", name = scope + "sc4", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,16,16], strides, padding, dim, d1, "K-RELU", name = scope + "sc5", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,16,16], strides, padding, dim, d1, "K-ABS", name = scope + "sc6", initializer=initializer, regularizer=regularizer)
  sd = ld.create_direct_sparse_to_dense(net, dim)
  sd_flat = tf.reshape(sd, [batch_size, total_size * 2])
  if train_labels != None:
    conv_out = tf.nn.dropout(sd_flat, 0.5, name="dropout")
  else:
    conv_out = sd_flat
  fc512 = tf.layers.dense(conv_out, 1024, name="dense2")
  fc10 = tf.layers.dense(fc512, num_classes, name="dense1")
  #if train:
  if train_labels != None:
    sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=train_labels, name = "softmax_loss"))
  else:
    sd_out = None
  p_sd_out = tf.nn.softmax(logits=fc10)
  return [sd_out, p_sd_out]

def model_modelnet_256(sparse_data, tensor_in_sizes, train_labels = None, num_classes = 10, scope = "mn256-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  dpooling_sizes = [2,2,2]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  d1 = 0.01
  net = ld.create_sparse_conv_layer(sd_converted, [3,3,3,1,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc1", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc2", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc3", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_pooling_layer(net, pooling_sizes, dim, 0.06)
  d2 = 0.03
  net = ld.create_sparse_conv_layer(net, [3,3,3,8,16], strides, padding, dim, d2, "K-RELU", name = scope + "sc4", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,16,16], strides, padding, dim, d2, "K-RELU", name = scope + "sc5", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,16,16], strides, padding, dim, d2, "K-RELU", name = scope + "sc6", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_pooling_layer(net, pooling_sizes, dim, 0.18)
  d3 = 0.07
  net = ld.create_sparse_conv_layer(net, [3,3,3,16,24], strides, padding, dim, d3, "K-RELU", name = scope + "sc7", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,24,24], strides, padding, dim, d3, "K-RELU", name = scope + "sc8", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [3,3,3,24,24], strides, padding, dim, d3, "K-RELU", name = scope + "sc9", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_pooling_layer(net, pooling_sizes, dim, 0.50)
  net = ld.create_direct_sparse_to_dense(net, dim)
  net = tf.reshape(net, [batch_size, 32, 32, 32, 24])
  #dense layers
  net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc10", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc11", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=32, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc12", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.max_pooling3d(inputs=net, pool_size=dpooling_sizes, strides=2, padding="same", name="dp1")
  net = tf.layers.conv3d(inputs=net, filters=40, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc13", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=40, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc14", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=40, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc15", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.max_pooling3d(inputs=net, pool_size=dpooling_sizes, strides=2, padding="same", name="dp2")
  net = tf.layers.conv3d(inputs=net, filters=48, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc16", kernel_initializer=initializer, kernel_regularizer=regularizer)
  net = tf.layers.conv3d(inputs=net, filters=48, kernel_size=[3, 3, 3], padding="same", activation=tf.nn.relu, name = scope + "sc17", kernel_initializer=initializer, kernel_regularizer=regularizer)
  n3 = net = tf.layers.conv3d(inputs=net, filters=48, kernel_size=[3, 3, 3], padding="same", name = scope + "sc18")
  if train_labels != None:
    net = tf.nn.dropout(net, 0.5, name="dropout")
  net = tf.reshape(net, [batch_size, -1])
  net = tf.layers.dense(net, 512)
  net = tf.layers.dense(net, num_classes)
  if train_labels == None:
    sd_out = None
  else:
    sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=train_labels, name = "softmax_loss"))
  p_sd_out = tf.nn.softmax(logits=net)
  return [sd_out, p_sd_out]

