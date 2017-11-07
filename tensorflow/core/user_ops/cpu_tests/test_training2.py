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
import pdb

tf.logging.set_verbosity(tf.logging.DEBUG)

def layer_to_sparse_tensor(layer):
  tensor = tf.SparseTensor(indices = layer.out_indices, values = layer.out_values, dense_shape = layer.out_shape)
  return tensor

def create_dense_and_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, dense_data, name = "conv"):
  with tf.variable_scope(name):
    [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -5, 10)
    sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
    f_ind_var = tf.Variable(filter1_ind, trainable=False, name="filter_indices")
    var_list.append(f_ind_var)
    f_w_var = tf.Variable(filter1_weights, trainable=True, name="filter_weights")
    var_list.append(f_w_var)
    f_sh_var = tf.Variable(filter1_sh, trainable=False, name="filter_shape")
    var_list.append(f_sh_var)
    dense_weights = tf.Variable(sp.sparse_to_dense(filter1_ind, filter1_weights, filter1_sh), trainable=True, name="dense_weights")
    var_list.append(dense_weights)
    sparse_conv = sc_module.sparse_tensor_sparse_kernel_dense_conv_kd(sparse_data.indices, sparse_data.values, sparse_data.dense_shape, f_ind_var, f_w_var, f_sh_var, strides, padding, dim, approx);
    conv = nn_ops.conv3d(dense_data, dense_weights, strides, padding)
    return [conv, sparse_conv]

def create_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, name = "conv"):
  with tf.variable_scope(name):
    [filter1_ind, filter1_weights, filter1_sh] = sp.createRandomSparseTensor(rho_filter, filter_in_sizes, -5, 10)
    sparse_filter_weights = tf.SparseTensor(indices=filter1_ind, values=filter1_weights, dense_shape=filter1_sh)
    f_ind_var = tf.Variable(filter1_ind, trainable=False, name="filter_indices")
    var_list.append(f_ind_var)
    f_w_var = tf.Variable(filter1_weights, trainable=True, name="filter_weights")
    var_list.append(f_w_var)
    f_sh_var = tf.Variable(filter1_sh, trainable=False, name="filter_shape")
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

#just a quick test, no nice code

filter_in_sizes_=[3, 3, 3, 1, 1] #[depth, height, width, in_channels, out_channels] 
stride=1
rho_data = 1
rho_filter=1
padding='SAME'
dim = 3
approx = False
res = 3
batch_size = 5
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]
nr_batchs = 10
batch_label_sizes = [5, 27]


filter_in_sizes = np.array(filter_in_sizes_, dtype=np.int64)
tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)



dense_data = tf.placeholder(tf.float32, shape=tensor_in_sizes, name="dense_placeholder")
sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")

if isinstance(stride, collections.Iterable):
   strides = [1] + list(stride) + [1]
else:
   strides = [1, stride, stride, stride, 1]

var_list = []

#initialize graph

sc1_ = create_sparse_conv_layer(filter_in_sizes, rho_filter, strides, padding, approx, dim, var_list, sparse_data, "sc1")
sc1 = layer_to_sparse_tensor(sc1_)
#sr1 = layer_to_sparse_tensor(create_sparse_relu_layer(sc1))
sr1 = layer_to_sparse_tensor(create_sparse_relu_layer_(sc1))
#sp1 = layer_to_sparse_tensor(create_sparse_pooling_layer(sr1, pooling_sizes, dim))
s_out = sr1


sd = sc_module.direct_sparse_to_dense(sparse_indices=s_out.indices, output_shape=s_out.dense_shape, sparse_values=s_out.values, default_value=0, validate_indices=False)

sd_flat = tf.reshape(sd, [batch_size, -1])

dense_labels = tf.placeholder(tf.float32, shape=sd_flat.shape, name="labels_placeholder")
sd_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sd_flat, labels=dense_labels, name = "softmax_loss"))


sd_train_op = tf.train.AdagradOptimizer(0.1)
pdb.run("sd_train =  sd_train_op.minimize(sd_loss)")
sd_grads = sd_train_op.compute_gradients(sd_loss)


config = tf.ConfigProto(
  device_count = {'GPU': 0}
)



initlocal = tf.variables_initializer(var_list)
initall = tf.global_variables_initializer()

#initialize variables
#create random training data
[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
random_dense_data = sp.sparse_to_dense(data_ind, data_val, data_sh)

[label_ind, label_val, label_sh] = sp.createRandomSparseTensor(1, batch_label_sizes)
random_dense_label = sp.sparse_to_dense(label_ind, label_val, label_sh)
with tf.Session(config=config) as sess:
  trainable = tf.trainable_variables()
  print("trainable: ", trainable)
  writer = tf.summary.FileWriter("/tmp/test", sess.graph)
  feed_dict={sparse_data: tf.SparseTensorValue(data_ind, data_val, data_sh), dense_data: random_dense_data, dense_labels: random_dense_label}
  sess.run(initlocal, feed_dict=feed_dict)
  sess.run(initall, feed_dict=feed_dict)
  
  for i in range(1, nr_batchs):
    print("batch nr: ", i)
    #create random training data
    [data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
    random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
    random_dense_data = sp.sparse_to_dense(data_ind, data_val, data_sh)

    [label_ind, label_val, label_sh] = sp.createRandomSparseTensor(1, batch_label_sizes)
    random_dense_label = sp.sparse_to_dense(label_ind, label_val, label_sh)

    #perform training
    sparse_result = sess.run(sd_train, feed_dict=feed_dict)
    #dense_result = sess.run(dd_train_op, feed_dict=feed_dict)

    sparse_grads = sess.run(sd_grads, feed_dict=feed_dict)
    print("sparse_grads: ", sparse_grads)
    rsc1 = tf.get_default_graph().get_tensor_by_name("sc1/filter_weights:0")
    print("filter weights: ", rsc1.eval())
    sparse_loss = sess.run(sd_loss, feed_dict=feed_dict)
    print("loss: ", sparse_loss)

  print("sparse result: ", sparse_result)
  #print("dense result: ", dense_result)


