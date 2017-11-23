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
  pointclouds_pl = tf.placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")
  labels_pl = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
  return pointclouds_pl, labels_pl

def get_model(dense_data, train_labels, is_training, tensor_in_sizes, num_classes = 10, scope = "mn16-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,2,2,2,1]
  batch_size = tensor_in_sizes[0]
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  ops = [None]*6
  net = tf.layers.conv3d(dense_data, 8, 3, (1,1,1), "SAME", activation=tf.nn.relu)
  net = tf.layers.conv3d(net, 8, 3, (1,1,1), "SAME", activation=tf.nn.relu)
  net = tf.layers.conv3d(net, 8, 3, (1,1,1), "SAME", activation=tf.nn.relu)
  net = tf.layers.max_pooling3d(net, (2,2,2), (2,2,2), "SAME")
  net = tf.layers.conv3d(net, 16, 3, (1,1,1), "SAME", activation=tf.nn.relu)
  net = tf.layers.conv3d(net, 16, 3, (1,1,1), "SAME", activation=tf.nn.relu)
  net = tf.layers.conv3d(net, 16, 3, (1,1,1), "SAME")
  sd_flat = tf.reshape(net, [batch_size, total_size * 2])
  conv_out =  tf.layers.dropout(sd_flat, 0.5, name="dropout", training=is_training)
  fc512 = tf.layers.dense(conv_out, 1024, name="dense2")
  fc10 = tf.layers.dense(fc512, num_classes, name="dense1")
  #if train:
  sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc10, labels=train_labels, name = "softmax_loss"))
  p_sd_out = tf.nn.softmax(logits=fc10)
  return [sd_out, p_sd_out, ops]
