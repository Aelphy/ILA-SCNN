
# coding: utf-8


import sys
import os
import time

import numpy as np

import direct_sparse_layer_definition as ld
import sys
import tensorflow as tf
import direct_sparse_regularizers as reg
import time
from direct_sparse_module import sparse_nn_ops as sc_module

pid = os.getpid()
print(pid)

def load_dataset():
  # We first define a download function, supporting both Python 2 and 3.
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

  # We then define functions for loading MNIST images and labels.
  # For convenience, they also download the requested files if needed.
  import gzip

  def load_mnist_images(filename):
    if not os.path.exists(filename):
      download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

  def load_mnist_labels(filename):
    if not os.path.exists(filename):
      download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

  # We can now download and read the training and test set images and labels.
  X_train = load_mnist_images('train-images-idx3-ubyte.gz')
  y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
  X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
  y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

  # We reserve the last 10000 training examples for validation.
  X_train, X_val = X_train[:-10000], X_train[-10000:]
  y_train, y_val = y_train[:-10000], y_train[-10000:]

  # We just return all the arrays in order, as expected in main().
  # (It doesn't matter how we do this as long as we can read them again.)
  return X_train, y_train, X_val, y_val, X_test, y_test



X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
(X_train * 255 < 50).sum() / (np.prod(X_train.shape) + 1e-7)

model_location = '/home/thackel/cnn_models/mnist'
pretrained_model = '/scratch/thackel/cnn_models/mnist_8'

dim = 5 
batch_size = 32
tensor_in_sizes_=[batch_size, 1, 28, 28, 1] #[batch, depth, height, width, in_channels]

num_classes = 10
batch_label_sizes = [batch_size, num_classes] 
tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)
sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name="sparse_placeholder")
dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name="labels_placeholder")

X_train = (X_train * 255).astype(np.uint8)
X_train[X_train<50] = 0

def model_mnist(sparse_data, tensor_in_sizes, train_labels = None, num_classes = 10, scope = "mn256-", initializer = None, regularizer = None):
  strides = [1,1,1,1,1]
  padding = "SAME"
  dim = 5
  pooling_sizes = [1,1,2,2,1]
  batch_size = tensor_in_sizes[0]
  print(batch_size)
  total_size = 1
  for i in range(1, len(tensor_in_sizes)): #skip batch size
    total_size = total_size * tensor_in_sizes[i]
  sd_converted = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
  d1 = 0.1
  net = ld.create_sparse_conv_layer(sd_converted, [1,3,3,1,8], strides, padding, dim, d1, "K-ABS", name = scope + "sc1", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [1,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc2", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [1,3,3,8,8], strides, padding, dim, d1, "K-RELU", name = scope + "sc3", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_pooling_layer(net, pooling_sizes, dim, 0.4)
  d2 = 0.3
  net = ld.create_sparse_conv_layer(net, [1,3,3,8,16], strides, padding, dim, d2, "K-ABS", name = scope + "sc4", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [1,3,3,16,16], strides, padding, dim, d2, "K-RELU", name = scope + "sc5", initializer=initializer, regularizer=regularizer)
  net = ld.create_sparse_conv_layer(net, [1,3,3,16,16], strides, padding, dim, d2, "K-ABS", name = scope + "sc6", initializer=initializer, regularizer=regularizer)
  net = ld.create_direct_sparse_to_dense(net, dim)
  net = tf.reshape(net, [batch_size, 1, 14, 14, 16])
  do = tf.layers.dropout(net, 0.5, name="dropout")
  net = tf.reshape(do, [batch_size, -1])
  net = tf.layers.dense(net, 512)
  net = tf.layers.dense(net, num_classes)
  if train_labels is None:
    sd_out = None
  else:
    sd_out = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=train_labels, name = "softmax_loss"))
  p_sd_out = tf.nn.softmax(logits=net)
  return [sd_out, p_sd_out, do]



print("initializing model")

[sd_loss, test_pred, do] = model_mnist(
    sparse_data=sparse_data, 
    tensor_in_sizes=tensor_in_sizes, 
    train_labels=dense_labels,
    num_classes=num_classes,
    regularizer = None
)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.01, global_step, 1000, 0.96, staircase=True)

sd_train_op = tf.train.AdagradOptimizer(learning_rate)
sd_train =  sd_train_op.minimize(sd_loss, global_step=global_step)




print("model initialized")


y_train_softmax = np.zeros((y_train.shape[0], 10))
y_test_softmax = np.zeros((y_test.shape[0], 10))
y_train_softmax[np.arange(y_train.shape[0]), y_train] = 1
y_test_softmax[np.arange(y_test.shape[0]), y_test] = 1

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batchsize]
    else:
      excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=20)

max_epochs = 50

with tf.Session() as sess:
  print("writing graph")
  trainable = tf.trainable_variables()
  print("trainable: ", trainable)
  sess.run(tf.global_variables_initializer())
  if len(pretrained_model) > 0:
    saver.restore(sess,pretrained_model)
  print("data initialized")


  
  for epoch in range(1, max_epochs):
    t1 = time.time()
    t_train1 = 0
    t_train2 = 0
    av_loss = 0
    batches = 0
    
    for batch in iterate_minibatches(X_train.reshape(-1, 1, 28, 28, 1), y_train_softmax, batch_size):
      #create random training data
      tt0 = time.time()

      feed_dict = {
        sparse_data: tf.SparseTensorValue(
          [cl for cl in zip(*[arr.astype(np.int64) for arr in batch[0].nonzero()])],
          batch[0][batch[0].nonzero()].astype(np.float32),
          batch[0].shape
        ),
        dense_labels: batch[1]
      }
      tt1 = time.time()
      #perform training
      [_, loss_val] = sess.run([sd_train, sd_loss], feed_dict=feed_dict)
      if (batches % 100) == 0:
        print("loss_val", loss_val)
      tt2 = time.time()
      av_loss = av_loss + loss_val
      batches = batches + 1

    t2 = time.time()
    av_loss = av_loss / batches
    print("epoch: ", epoch)
    print("average loss: ", av_loss)
    print("time all: ", t2 - t1)
    print("time train: ", t_train2 - t_train1)
    saver.save(sess, model_location + "_" + str(epoch))


  do.training = False

  sum_correct = 0
  batches = 0

  correct_prediction = tf.equal(tf.argmax(test_pred, 1), tf.argmax(dense_labels, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  for batch in iterate_minibatches(X_test.reshape(-1, 1, 28, 28, 1), y_test_softmax, batch_size):
    feed_dict = {
      sparse_data: tf.SparseTensorValue(
        [cl for cl in zip(*[arr.astype(np.int64) for arr in batch[0].nonzero()])],
        batch[0][batch[0].nonzero()].astype(np.float32),
        batch[0].shape
      ),
      dense_labels: batch[1]
    }
    accuracy_batch = sess.run(accuracy, feed_dict=feed_dict)
    sum_correct = sum_correct + accuracy_batch * batch_size
    batches = batches + 1
  accuracy_all = sum_correct / (batches * batch_size)
  print("accuracy", accuracy_all)


