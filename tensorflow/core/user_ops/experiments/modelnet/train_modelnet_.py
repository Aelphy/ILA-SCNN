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
from sparse_module import sparse_nn_ops as sc_module


#just a quick test, no nice code

data_location = '/home/thackel/Desktop/ModelNet10'
pretrained_model = ''
model_location = '/home/thackel/cnn_models/modelnet10_8'
learning_rate = 0.0001
dim = 3
approx = False
res = 8
rho_data = 0.01
batch_size = 32
tensor_in_sizes_=[batch_size, res, res, res, 1] #[batch, depth, height, width, in_channels]
pooling_sizes = [1,2,2,2,1]
reader = ModelnetReader(data_location, res, 0, batch_size)
num_classes = reader.getNumClasses()
batch_label_sizes = [batch_size, num_classes]
max_epochs = 1000

tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)

sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")
dense_data = tf.placeholder(tf.float32, shape=tensor_in_sizes, name="dense_placeholder")


var_list = []

#initialize graph

dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")
[sd_loss, p_sd_loss] = models.model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = True, train_labels = dense_labels, num_classes = num_classes, approx = False)
sd_train_op = tf.train.AdagradOptimizer(learning_rate)
sd_train =  sd_train_op.minimize(sd_loss)
sd_grads = sd_train_op.compute_gradients(sd_loss)


[sda_loss, p_sda_loss] = models.model_modelnet10_8(sparse_data, tensor_in_sizes, var_list, train = True, train_labels = dense_labels, num_classes = num_classes, approx = True, scope = "a/")
sda_train_op = tf.train.AdagradOptimizer(learning_rate)
sda_train =  sd_train_op.minimize(sda_loss)
sda_grads = sd_train_op.compute_gradients(sda_loss)

[dd_loss, p_dd_loss] = models.dense_model_modelnet10_8(dense_data, tensor_in_sizes, var_list, train = True, train_labels = dense_labels, num_classes = num_classes, approx = approx)
dd_train_op = tf.train.AdagradOptimizer(learning_rate)
dd_train =  sd_train_op.minimize(dd_loss)
dd_grads = sd_train_op.compute_gradients(dd_loss)

config = tf.ConfigProto(
  device_count = {'GPU': 0}
)


saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=50)

initlocal = tf.variables_initializer(var_list)
initall = tf.global_variables_initializer()

#initialize variables
#create random training data
[data_ind, data_val, data_sh] = sp.createRandomSparseTensor(rho_data, tensor_in_sizes)
random_sparse_data = tf.SparseTensor(indices=data_ind, values=data_val, dense_shape=data_sh)
dense_data_ = sp.sparse_to_dense(data_ind, data_val, data_sh)

[label_ind, label_val, label_sh] = sp.createRandomSparseTensor(1, batch_label_sizes)
random_dense_label = sp.sparse_to_dense(label_ind, label_val, label_sh)
with tf.Session(config=config) as sess:
  trainable = tf.trainable_variables()
  print("trainable: ", trainable)
  writer = tf.summary.FileWriter("/tmp/test", sess.graph)
  feed_dict={sparse_data: tf.SparseTensorValue(data_ind, data_val, data_sh), dense_labels: random_dense_label, dense_data: dense_data_}
  sess.run(initlocal, feed_dict=feed_dict)
  sess.run(initall, feed_dict=feed_dict)
  #set initial filter weights to same value
  sci = tf.get_default_graph().get_tensor_by_name("sc1/filter_indices:0")
  scv = tf.get_default_graph().get_tensor_by_name("sc1/filter_weights:0")
  scs = tf.get_default_graph().get_tensor_by_name("sc1/filter_shape:0")
  dw = [var for var in tf.global_variables() if var.op.name=="dc1/dense_weights"][0]
  sess.run(dw.assign(sp.sparse_to_dense(sci.eval(), scv.eval(), scs.eval())))

  saci = [var for var in tf.global_variables() if var.op.name=="a/sc1/filter_indices"][0]
  sacv = [var for var in tf.global_variables() if var.op.name=="a/sc1/filter_weights"][0]
  sacs = [var for var in tf.global_variables() if var.op.name=="a/sc1/filter_shape"][0]
  sess.run(saci.assign(sci))
  sess.run(sacv.assign(scv))
  sess.run(sacs.assign(scs))

  if len(pretrained_model) > 0:
    saver.restore(sess,pretrained_model)
  for epoch in range(1, max_epochs):
    #train
    reader = ModelnetReader(data_location, res, 0, batch_size, train=True)
    reader.init()
    reader.start()
    has_data = True
    av_loss = 0
    d_av_loss = 0
    a_av_loss = 0
    batches = 0
    t1 = time.time()
    while has_data:
      #create random training data
      [batch, has_data] = reader.next_batch()
      reader.start()
      values_ = np.array(batch[1], dtype=np.float32)
      indices_ = np.array(batch[0], dtype =np.int64)
      dense_data_ = sp.sparse_to_dense(indices_, values_, batch[2])
      feed_dict={sparse_data: tf.SparseTensorValue(indices_, values_, batch[2]), dense_labels: batch[3], dense_data: dense_data_}

      #perform training
      [_, loss_val] = sess.run([sd_train, sd_loss], feed_dict=feed_dict)
      [_, a_loss_val] = sess.run([sda_train, sda_loss], feed_dict=feed_dict)
      [_, d_loss_val] = sess.run([dd_train, dd_loss], feed_dict=feed_dict)
      av_loss = av_loss + loss_val
      a_av_loss = a_av_loss + a_loss_val
      d_av_loss = d_av_loss + d_loss_val
      batches = batches + 1
    t2 = time.time()
    av_loss = av_loss / batches
    a_av_loss = a_av_loss / batches
    d_av_loss = d_av_loss / batches
    print("epoch: ", epoch)
    print("average sparse loss: ", av_loss)
    print("average dense loss: ", d_av_loss)
    print("average approx sparse loss: ", a_av_loss)
    print("time: ", t2 - t1)
    saver.save(sess, model_location + "_" + str(epoch))

    #predict on test set
    reader = ModelnetReader(data_location, res, 0, batch_size, train=False)
    reader.init()
    reader.start()
    has_data = True
    all_conf_mat = tf.zeros([num_classes, num_classes], dtype=tf.int32)
    d_all_conf_mat = tf.zeros([num_classes, num_classes], dtype=tf.int32)
    as_all_conf_mat = tf.zeros([num_classes, num_classes], dtype=tf.int32)
    

    while has_data:
      t1 = time.time()
      [batch, has_data] = reader.next_batch()
      reader.start()
      t2 = time.time()
      print("time: ", t2 - t1) 
      values_ = np.array(batch[1], dtype=np.float32)
      indices_ = np.array(batch[0], dtype =np.int64)
      dense_data_ = sp.sparse_to_dense(indices_, values_, batch[2])
      feed_dict={sparse_data: tf.SparseTensorValue(indices_, values_, batch[2]), dense_labels: batch[3], dense_data: dense_data_}
      #perform training
      [loss_val, d_loss_val, as_loss_val] = sess.run([p_sd_loss, p_dd_loss, p_sda_loss], feed_dict=feed_dict)
      predictions = sess.run(tf.argmax(loss_val, axis=1))
      actual = sess.run(tf.argmax(batch[3], axis=1))
      all_conf_mat = all_conf_mat  + tf.confusion_matrix(actual, predictions, num_classes=num_classes)
      d_predictions = sess.run(tf.argmax(d_loss_val, axis=1))
      d_all_conf_mat = d_all_conf_mat  + tf.confusion_matrix(actual, d_predictions, num_classes=num_classes)
      as_predictions = sess.run(tf.argmax(as_loss_val, axis=1))
      as_all_conf_mat = as_all_conf_mat  + tf.confusion_matrix(actual, as_predictions, num_classes=num_classes)
    [res_conf_mat, [iou, aiou, oa]] = sess.run([all_conf_mat, sc_module.evaluate_confusion_matrix(all_conf_mat)])
    print("sparse confusion matrix: ", res_conf_mat)
    print("sparse iou:", iou)
    print("sparse aiou:", aiou)
    print("sparse oa:", oa) 

    [d_res_conf_mat, [d_iou, d_aiou, d_oa]] = sess.run([d_all_conf_mat, sc_module.evaluate_confusion_matrix(d_all_conf_mat)])
    print("dense confusion matrix: ", d_res_conf_mat)
    print("dense iou:", d_iou)
    print("dense aiou:", d_aiou)
    print("dense oa:", d_oa)

    [as_res_conf_mat, [as_iou, as_aiou, as_oa]] = sess.run([as_all_conf_mat, sc_module.evaluate_confusion_matrix(as_all_conf_mat)])
    print("approx confusion matrix: ", as_res_conf_mat)
    print("approx iou:", as_iou)
    print("approx aiou:", as_aiou)
    print("approx oa:", as_oa)

    arr = np.array([[epoch, av_loss, d_av_loss, a_av_loss, iou[0], aiou[0], oa[0], d_iou[0], d_aiou[0], d_oa[0], as_iou[0], as_aiou[0], as_oa[0]]])
    f_handle = file("res_training.txt", 'a')
    np.savetxt(f_handle, arr, fmt='%.18g', delimiter=' ', newline=os.linesep)
    f_handle.close()
