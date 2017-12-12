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
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import random
import numpy as np
import time
import direct_sparse_regularizers as reg
import sparse_tools as sp
import os
from tensorflow.python import debug as tf_debug
import modelnet_models as models
from read_modelnet_models import ModelnetReader
#just a quick test, no nice code

res = 256
data_location = '/scratch/thackel/ModelNet10'
pretrained_model = ''
model_location = '/home/thackel/cnn_models/modelnet10_256'
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step, 1000, 0.96, staircase=True)
initializer = tf.truncated_normal_initializer(0, 0.1)

rho_data = 1 / (3 * res)
batch_size = 32
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
print("started construction data reader")
reader = ModelnetReader(data_location, res, 0, batch_size, preprocess=True)
print("data reader constructed")
num_classes = reader.getNumClasses()
print("number of classes", num_classes)
batch_label_sizes = [batch_size, num_classes]
max_epochs = 1000
dim = 5
#regularizer =  reg.biased_l2_regularizer(0.005, -0.005)
regularizer =  None

tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)
sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")

#initialize graph
dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
print("started model generation")
[sd_loss, test_loss] = models.model_modelnet_res(res, sparse_data, tensor_in_sizes, train_labels = dense_labels, num_classes = num_classes, initializer = initializer, regularizer = regularizer)
print("model generated")
sd_train_op = tf.train.AdagradOptimizer(learning_rate)
sd_train =  sd_train_op.minimize(sd_loss, global_step=global_step)
sd_grads = sd_train_op.compute_gradients(sd_loss)

config = tf.ConfigProto(
#device_count = {'GPU': 0}
)

config.gpu_options.per_process_gpu_memory_fraction = 0.8

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=100)
initall = tf.global_variables_initializer()

#initialize variables
#create random training data
[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)

[label_ind, label_val, label_sh] = sp.createRandomSparseTensor(1, batch_label_sizes)
random_dense_label = sp.sparse_to_dense(label_ind, label_val, label_sh)
with tf.Session(config=config) as sess:
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  print("writing graph")
  writer = tf.summary.FileWriter("/tmp/test", sess.graph)
  trainable = tf.trainable_variables()
  print("trainable: ", trainable)
  feed_dict={sparse_data: tf.SparseTensorValue(data_ind, data_val, data_sh), dense_labels: random_dense_label}
  sess.run(initall, feed_dict=feed_dict)
  print("data initialized")
  if len(pretrained_model) > 0:
    saver.restore(sess,pretrained_model)
  for epoch in range(1, max_epochs):
    reader.init()
    has_data = True
    av_loss = 0
    batches = 0
    t1 = time.time()
    t_train = 0
    t_data_read = 0
    while True:
      #create random training data
      tt0 = time.time()
      [batch, has_data] = reader.next_batch()
      if has_data == False:
        break
      values_ = np.array(batch[1], dtype=np.float32)
      indices_ = np.array(batch[0], dtype =np.int64)
      #print(indices_, values_, batch[2])
      tt1 = time.time()
      feed_dict={sparse_data: tf.SparseTensorValue(indices_, values_, batch[2]), dense_labels: batch[3]}
      #perform training
      [_, loss_val] = sess.run([sd_train, sd_loss], feed_dict=feed_dict)
      tt2 = time.time()
      if batches < 20 or (batches % 20) == 0:
        print("loss val: ", loss_val)
      av_loss = av_loss + loss_val
      batches = batches + 1
      t_train = t_train + tt2 - tt1
      t_data_read = t_data_read + tt1 - tt0
    t2 = time.time()
    av_loss = av_loss / batches
    print("epoch: ", epoch)
    print("average loss: ", av_loss)
    print("time data read: ", t_data_read)
    print("time train: ", t_train)
    saver.save(sess, model_location + "_" + str(epoch))
  saver.save(sess, model_location)
