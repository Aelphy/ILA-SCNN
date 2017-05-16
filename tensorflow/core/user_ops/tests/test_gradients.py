from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
import tensorflow as tf
import random
import numpy as np
import time
import sparse_tools as sp
import os
import sparse_ops #registration of gradients
from tensorflow.python import debug as tf_debug
import layer_definition as ld #sparse layers
from sparse_module import sparse_nn_ops as sc_module #sparse operations
import pdb

#just a quick test, no nice code

filter_in_sizes_=[3, 3, 3, 1, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
rho_data = 0.03
rho_filter=1
padding='SAME'
dim = 3
approx = False
res = 3
batch_size = 5
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]

filter_in_sizes = np.array(filter_in_sizes_, dtype=np.int64)
tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)

#dense_data = tf.placeholder(tf.float32, shape=tensor_in_sizes, name="dense_placeholder")
#sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")

[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
random_dense_data = sp.sparse_to_dense(data_ind, data_val, data_sh)

dense_data = random_dense_data
sparse_data = random_sparse_data

if isinstance(stride, collections.Iterable):
   strides = [1] + list(stride) + [1]
else:
   strides = [1, stride, stride, stride, 1]

var_list = []

#initialize graph
#sc1_ = ld.create_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, "sc1")
#sc1 = ld.layer_to_sparse_tensor(sc1_)

sc1_ = ld.create_sparse_pooling_layer(sparse_data, pooling_sizes, dim)
sc1 = ld.layer_to_sparse_tensor(sc1_)
#sc1 = sc1_

#sd = sc_module.direct_sparse_to_dense(sparse_indices=sc1.out_indices, output_shape=sc1.out_shape, sparse_values=sc1.out_values, default_value=0, validate_indices=False)
sd = ld.create_direct_sparse_to_dense(sc1)

sd_flat = tf.reshape(sd, [batch_size, -1])
#dd_flat = tf.reshape(dc1, [batch_size, -1])

dense_labels = tf.placeholder(tf.float32, shape=sd_flat.shape, name="labels_placeholder")
#sd_loss = tf.Variable(tf.losses.softmax_cross_entropy(dense_labels, sd_flat), name = "sparse_loss")
#dd_loss = tf.Variable(tf.losses.softmax_cross_entropy(dense_labels, dd_flat), name = "dense_loss")

#sd_train_op = tf.train.GradientDescentOptimizer(0.01).minimize(sd_loss)

grads =tf.gradients(sd, list(sc1_), name='sc1', colocate_gradients_with_ops=True, gate_gradients=True)


#opt = tf.train.AdagradOptimizer(0.1)
#grads = opt.compute_gradients(dd_loss)
#apply_transform_op = opt.apply_gradients(grads)


config = tf.ConfigProto(
  device_count = {'GPU': 0}
)

initlocal = tf.variables_initializer(var_list)
initall = tf.global_variables_initializer()

#initialize variables
#create random training data


random_dense_label = []
#[label_ind, label_val, label_sh] = sp.createRandomSparseTensor(1, dd_flat.get_shape().as_list())
#random_dense_label = sp.sparse_to_dense(label_ind, label_val, label_sh)
with tf.Session(config=config) as sess:
  trainable = tf.trainable_variables()
  print("trainable: ", trainable)
  writer = tf.summary.FileWriter("/tmp/test", sess.graph)
  #feed_dict={sparse_data: tf.SparseTensorValue(data_ind, data_val, data_sh), dense_data: random_dense_data, dense_labels: random_dense_label}
  #feed_dict={dense_labels: random_dense_label}
  feed_dict={}
  
  sess.run(initlocal)
  sess.run(initall)
  pdb.run("sparse_result = sess.run(grads)")
  #rsc1 = tf.get_default_graph().get_tensor_by_name("sc1/filter_weights:0")
  #print("filter weights: ", rsc1.eval())

  print("sparse result: ", sparse_result)
  #print("dense result: ", dense_result)


