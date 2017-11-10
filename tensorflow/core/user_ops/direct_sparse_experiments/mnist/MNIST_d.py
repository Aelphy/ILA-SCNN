
# coding: utf-8

# In[ ]:


import sys
import os
import time

import numpy as np
import tensorflow as tf

import direct_sparse_layer_definition as ld
import direct_sparse_regularizers as reg

from direct_sparse_module import sparse_nn_ops as sc_module
from load_mnist_dataset import load_dataset


# In[ ]:


def model_mnist(
    sparse_data,
    tensor_in_sizes,
    keep_prob,
    train_labels=None,
    num_classes=10,
    scope='mn256-',
    initializer=None,
    regularize=True,
    d1=0.1,
    d2=0.3,
    d3=0.4
):
    dim = 5
    strides = [1,1,1,1,1]
    padding = 'SAME'
    pooling_sizes = [1,1,2,2,1]
    batch_size = tensor_in_sizes[0]
    total_size = np.prod(tensor_in_sizes)

    net = {}

    net['sd_converted'] = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
    net['conv1_1'] = ld.create_sparse_conv_layer(
        net['sd_converted'],
        [1,3,3,1,8],
        strides,
        padding,
        dim,
        d1,
        'K-ABS',
        name=scope + 'sc1',
        initializer=initializer,
        regularize=regularize
    )
    net['conv1_2'] = ld.create_sparse_conv_layer(
        net['conv1_1'],
        [1,3,3,8,8],
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc2',
        initializer=initializer,
        regularize=regularize
    )
    net['conv1_3'] = ld.create_sparse_conv_layer(
        net['conv1_2'],
        [1,3,3,8,8],
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc3',
        initializer=initializer,
        regularize=regularize
    )
    net['pool1'] = ld.create_sparse_pooling_layer(net['conv1_3'], pooling_sizes, dim, d3)
    net['conv2_1'] = ld.create_sparse_conv_layer(
        net['pool1'],
        [1,3,3,8,16],
        strides,
        padding,
        dim,
        d2,
        'K-ABS',
        name=scope + 'sc4',
        initializer=initializer,
        regularize=regularize
    )
    net['conv2_2'] = ld.create_sparse_conv_layer(
        net['conv2_1'],
        [1,3,3,16,16],
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc5',
        initializer=initializer,
        regularize=regularize
    )
    net['conv2_3'] = ld.create_sparse_conv_layer(
        net['conv2_2'],
        [1,3,3,16,16],
        strides,
        padding,
        dim,
        d2,
        'K-ABS',
        name=scope + 'sc6',
        initializer=initializer,
        regularize=regularize
    )
    net['sparse_to_dense'] = ld.create_direct_sparse_to_dense(net['conv2_3'], dim)
    net['dense_reshaped'] = tf.reshape(net['sparse_to_dense'], [batch_size, 1, 14, 14, 16])
    net['do'] = tf.layers.dropout(net['dense_reshaped'], keep_prob, training=True, name='dropout')
    net['do_reshaped'] = tf.reshape(net['do'], [batch_size, -1])
    net['dense1'] = tf.layers.dense(net['do_reshaped'], 512)
    net['dense2'] = tf.layers.dense(net['dense1'], num_classes)

    predictions = {
      'classes': tf.argmax(net['dense2'], axis=1),
      'probabilities': tf.nn.softmax(net['dense2'])
    }

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=train_labels,
        logits=net['dense2']
    )

    accuracy = tf.metrics.accuracy(tf.argmax(train_labels, axis=1), predictions['classes'])

    return loss, predictions, accuracy, net


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


# In[ ]:


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

X_train = (X_train * 255).astype(np.uint8)
X_train[X_train<50] = 0

X_val = (X_val * 255).astype(np.uint8)
X_val[X_val<50] = 0

X_test = (X_test * 255).astype(np.uint8)
X_test[X_test<50] = 0

y_train_softmax = np.zeros((y_train.shape[0], 10))
y_train_softmax[np.arange(y_train.shape[0]), y_train] = 1

y_val_softmax = np.zeros((y_val.shape[0], 10))
y_val_softmax[np.arange(y_val.shape[0]), y_val] = 1

y_test_softmax = np.zeros((y_test.shape[0], 10))
y_test_softmax[np.arange(y_test.shape[0]), y_test] = 1

dim = 5
batch_size = 100
tensor_in_sizes_=[batch_size, 1, 28, 28, 1] #[batch, depth, height, width, in_channels]

num_classes = 10
batch_label_sizes = [batch_size, num_classes]
tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)
sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name='sparse_placeholder')
dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name='labels_placeholder')
keep_prob = tf.placeholder(tf.float32)


# In[ ]:


for d1 in [0.03, 0.035, 0.04, 0.045, 0.065, 0.075, 0.085, 0.095, 0.15, 0.2]:
    print('===============================================================')
    print('===============================================================')

    print(d1)

    print('===============================================================')
    print('===============================================================')

    with tf.Session() as sess:
        loss, predictions, accuracy, net = model_mnist(
            sparse_data,
            tensor_in_sizes,
            keep_prob,
            dense_labels,
            num_classes,
            scope='mn256_d{}-'.format(d1),
            regularize=False,
            d1 = d1,
            d2 = 2*d1,
            d3 = 4*d1
        )

        print('initializing model')

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('data initialized')

        print('model initialized')

        num_epochs = 10

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_acc = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(X_train.reshape(-1, 1, 28, 28, 1), y_train_softmax, batch_size):
                feed_dict = {
                    sparse_data: tf.SparseTensorValue(
                      [cl for cl in zip(*[arr.astype(np.int64) for arr in batch[0].nonzero()])],
                      batch[0][batch[0].nonzero()].astype(np.float32),
                      batch[0].shape
                    ),
                    dense_labels: batch[1],
                    keep_prob: 0.5
                }

                _, train_err_batch, train_acc_batch = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

                train_err += train_err_batch
                train_acc += train_acc_batch[0]
                train_batches += 1

            # And a full pass over the validation data:
            val_acc = 0
            val_batches = 0

            for batch in iterate_minibatches(X_val.reshape(-1, 1, 28, 28, 1), y_val_softmax, batch_size):
                feed_dict = {
                    sparse_data: tf.SparseTensorValue(
                        [cl for cl in zip(*[arr.astype(np.int64) for arr in batch[0].nonzero()])],
                        batch[0][batch[0].nonzero()].astype(np.float32),
                        batch[0].shape
                    ),
                    dense_labels: batch[1],
                    keep_prob: 1
                }

                val_acc_batch = sess.run(accuracy, feed_dict=feed_dict)

                val_acc += val_acc_batch[0]
                val_batches += 1

            print('Epoch {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, time.time() - start_time))
            print('  training loss (in-iteration):\t{:.6f}'.format(train_err / train_batches))
            print('  train accuracy:\t\t{:.2f} %'.format(train_acc / train_batches * 100))
            print('  validation accuracy:\t\t{:.2f} %'.format(val_acc / val_batches * 100))
            print('  removed {} out of {} weights'.format(removed_weights, weights_total))

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=1)
        saver.save(sess, './' + '_d1_' + str(dr))

