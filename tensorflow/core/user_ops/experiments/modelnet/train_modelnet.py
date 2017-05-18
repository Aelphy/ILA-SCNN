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
import modelnet_models as models
from read_modelnet_models import ModelnetReader

#just a quick test, no nice code

data_location = '/home/thackel/Desktop/ModelNet10'
model_location = '/tmp/modelnet10_8'
learning_rate = 0.01
dim = 3
approx = True
res = 8
rho_data = 1. / res
batch_size = 32
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]
batch_label_sizes = [batch_size, 10]
max_epochs = 200


tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)

sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")


var_list = []

#initialize graph

dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
sd_loss = models.model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = True, train_labels = dense_labels, approx = approx)
sd_train_op = tf.train.AdagradOptimizer(learning_rate)
sd_train =  sd_train_op.minimize(sd_loss)
sd_grads = sd_train_op.compute_gradients(sd_loss)

config = tf.ConfigProto(
  device_count = {'GPU': 0}
)


saver = tf.train.Saver(var_list=tf.trainable_variables())

initlocal = tf.variables_initializer(var_list)
initall = tf.global_variables_initializer()

#initialize variables
#create random training data
[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)

[label_ind, label_val, label_sh] = sp.createRandomSparseTensor(1, batch_label_sizes)
random_dense_label = sp.sparse_to_dense(label_ind, label_val, label_sh)
with tf.Session(config=config) as sess:
  trainable = tf.trainable_variables()
  print("trainable: ", trainable)
  writer = tf.summary.FileWriter("/tmp/test", sess.graph)
  feed_dict={sparse_data: tf.SparseTensorValue(data_ind, data_val, data_sh), dense_labels: random_dense_label}
  sess.run(initlocal, feed_dict=feed_dict)
  sess.run(initall, feed_dict=feed_dict)
  for epoch in range(1, max_epochs):
    reader = ModelnetReader(data_location, res, 8, batch_size)
    reader.init()
    reader.start()
    has_data = True
    av_loss = 0
    batches = 0
    while has_data:
      #create random training data
      t1 = time.time()
      [batch, has_data] = reader.next_batch()
      reader.start()
      t2 = time.time()
      print("time: ", t2 - t1)
      values_ = np.array(batch[1], dtype=np.float32)
      indices_ = np.array(batch[0], dtype =np.int64)
      feed_dict={sparse_data: tf.SparseTensorValue(indices_, values_, batch[2]), dense_labels: batch[3]}

      #perform training
      [_, loss_val] = sess.run([sd_train, sd_loss], feed_dict=feed_dict)
      av_loss = av_loss + loss_val
      batches = batches + 1
      '''sparse_grads = sess.run(sd_grads, feed_dict=feed_dict)
      print("sparse_grads: ", sparse_grads)
      rsc1 = tf.get_default_graph().get_tensor_by_name("sc1/filter_weights:0")
      print("filter weights: ", rsc1.eval())
      sparse_loss = sess.run(sd_loss, feed_dict=feed_dict)
      print("loss: ", sparse_loss)'''
      print("loss val: ", loss_val)
    av_loss = av_loss / batches
    print("average loss: ", av_loss)
    saver.save(sess, model_location + str(epoch))
  saver.save(sess, model_location)



