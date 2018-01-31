import os
import sys
import time

import numpy as np
import tensorflow as tf

import direct_sparse_layer_definition as ld
import direct_sparse_regularizers as reg

from direct_sparse_module import sparse_nn_ops as sc_module
from load_mnist_dataset import load_dataset




def model_mnist(
    sparse_data, 
    tensor_in_sizes,
    train_labels=None,
    num_classes=10,
    scope='mn256-',
    initializer=None,
    d1=0.1,
    d2=0.3,
    d3=0.4,
    rscale=0.02,
    max_bias=0.1
):
    dim = 5
    strides = [1,1,1,1,1]
    padding = 'SAME'
    pooling_sizes = [1,1,2,2,1]
    batch_size = tensor_in_sizes[0]
    total_size = np.prod(tensor_in_sizes)
    
    net = {}
    ops = [None]*6
    
    tmp_tin = tensor_in_sizes
    
    net['sd_converted'] = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
    net['conv1_1'], tmp_tin, ops[0] = ld.create_sparse_conv_layer_reg(
        net['sd_converted'],
        [1,3,3,1,8],
        tmp_tin,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc1',
        initializer=initializer,
        scale = rscale,
        bias_offset = max_bias
    )
    net['conv1_2'], tmp_tin, ops[1] = ld.create_sparse_conv_layer_reg(
        net['conv1_1'],
        [1,3,3,8,8],
        tmp_tin,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc2',
        initializer=initializer,
        scale = rscale,
        bias_offset = max_bias
    )
    net['conv1_3'], tmp_tin, ops[2] = ld.create_sparse_conv_layer_reg(
        net['conv1_2'],
        [1,3,3,8,8],
        tmp_tin,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc3',
        initializer=initializer,
        scale = rscale,
        bias_offset = max_bias
    )
    net['pool1'], tmp_tin = ld.create_sparse_pooling_layer(net['conv1_3'], pooling_sizes, tmp_tin, dim, d3)
    net['conv2_1'], tmp_tin, ops[3] = ld.create_sparse_conv_layer_reg(
        net['pool1'],
        [1,3,3,8,16],
        tmp_tin,
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc4',
        initializer=initializer,
        scale = rscale,
        bias_offset = max_bias
    )
    net['conv2_2'], tmp_tin, ops[4] = ld.create_sparse_conv_layer_reg(
        net['conv2_1'],
        [1,3,3,16,16],
        tmp_tin,
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc5',
        initializer=initializer,
        scale = rscale,
        bias_offset = max_bias
    )
    net['conv2_3'], tmp_tin, ops[5] = ld.create_sparse_conv_layer_reg(
        net['conv2_2'],
        [1,3,3,16,16],
        tmp_tin,
        strides,
        padding,
        dim,
        d2,
        'K-ABS',
        name=scope + 'sc6',
        initializer=initializer,
        scale = rscale,
        bias_offset = max_bias
    )
    net['sparse_to_dense'] = ld.create_direct_sparse_to_dense(net['conv2_3'], dim)
    net['dense_reshaped1'] = tf.reshape(net['sparse_to_dense'], [batch_size, 1, 14, 14, 16])
    net['dense_reshaped2'] = tf.reshape(net['dense_reshaped1'], [batch_size, -1])
    net['dense1'] = tf.layers.dense(net['dense_reshaped2'], 512)
    net['dense2'] = tf.layers.dense(net['dense1'], num_classes)

    predictions = {
      'classes': tf.argmax(net['dense2'], axis=1),
      'probabilities': tf.nn.softmax(net['dense2'])
    }
    
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=train_labels,
        logits=net['dense2']
    )
    
    loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    accuracy = tf.metrics.accuracy(tf.argmax(train_labels, axis=1), predictions['classes'])
    
    return loss, predictions, accuracy, net, ops


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
batch_size = 32
tensor_in_sizes_=[batch_size, 1, 28, 28, 1] #[batch, depth, height, width, in_channels]

num_classes = 10
batch_label_sizes = [batch_size, num_classes]
tensor_in_sizes = np.array(tensor_in_sizes_, dtype=np.int64)
sparse_data = tf.sparse_placeholder(tf.float32, shape=tensor_in_sizes, name='sparse_placeholder')
dense_labels = tf.placeholder(tf.float32, shape=batch_label_sizes, name='labels_placeholder')


# In[ ]:


with open('thr_experiment2.log', 'wb') as f:
    for pruning_thr in [5e-3]:
        print('===============================================================')
        print('===============================================================')

        print(pruning_thr)
        f.write(str(pruning_thr)+'\n')

        print('===============================================================')
        print('===============================================================')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        with tf.Session(config=config) as sess:
            print('initializing model')
            loss, predictions, accuracy, net, ops = model_mnist(
                sparse_data,
                tensor_in_sizes,
                dense_labels,
                num_classes,
                scope='mn256_thr{}-'.format(pruning_thr),
                d1 = 0.1,
                d2 = 0.2,
                d3 = 0.4
            )

            optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print('data and model are initialized')
            f.write('data and model are initialized'+'\n')

            num_epochs = 30

            kernels = {}
            to_remove = {}

            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                if 'filter' in var.name and 'Adagrad' not in var.name:
                    parts = var.name.split('/')

                    if parts[0] not in kernels:
                        kernels[parts[0]] = {}

                    if 'filter_indices' in parts[1]:
                        kernels[parts[0]]['filter_indices'] = var
                    if 'filter_shape' in parts[1]:
                        kernels[parts[0]]['filter_shape'] = var
                    if 'filter_channel_mapping' in parts[1]:
                        kernels[parts[0]]['filter_channel_mapping'] = var
                    if 'filter_values' in parts[1]:
                        kernels[parts[0]]['filter_values'] = var
            for epoch in range(num_epochs):
                # In each epoch, we do a full pass over the training data:
                train_err = 0
                train_acc = 0
                train_batches = 0
                density_layer3 = 0
                start_time = time.time()

                for batch in iterate_minibatches(X_train.reshape(-1, 1, 28, 28, 1), y_train_softmax, batch_size):
                    feed_dict = {
                        sparse_data: tf.SparseTensorValue(
                          [cl for cl in zip(*[arr.astype(np.int64) for arr in batch[0].nonzero()])],
                          batch[0][batch[0].nonzero()].astype(np.float32),
                          batch[0].shape
                        ),
                        dense_labels: batch[1]
                    }

                    _, train_err_batch, train_acc_batch, n1 = sess.run([train_op, loss, accuracy, net['conv1_3']], feed_dict=feed_dict)
                    sess.run(ops, feed_dict=feed_dict)
                    
                    density_layer3 += np.mean(n1.out_channel_densities)
                    #print(np.mean(n1.out_channel_densities), n1.out_channel_densities)

                    train_err += train_err_batch
                    train_acc += train_acc_batch[0]
                    train_batches += 1

                training_time = time.time()

                print('Epoch {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, training_time - start_time))
                print('  train loss    :\t\t{:.6f}'.format(train_err / train_batches))
                print('  train accuracy:\t\t{:.2f} %'.format(train_acc / train_batches * 100))
                print('  average density at conv layer 3:\t\t{:.2f} %'.format(density_layer3 / train_batches * 100))
                f.write('Epoch {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, training_time - start_time)+'\n')
                f.write('  train loss    :\t\t{:.6f}'.format(train_err / train_batches)+'\n')
                f.write('  train accuracy:\t\t{:.2f} %'.format(train_acc / train_batches * 100)+'\n')
                f.write('  density:\t\t{:.2f} %'.format(density_layer3 / train_batches * 100)+'\n')

                        
                # And a full pass over the validation data:
                val_err = 0
                val_acc = 0
                val_batches = 0

                for batch in iterate_minibatches(X_val.reshape(-1, 1, 28, 28, 1), y_val_softmax, batch_size):
                    feed_dict = {
                        sparse_data: tf.SparseTensorValue(
                            [cl for cl in zip(*[arr.astype(np.int64) for arr in batch[0].nonzero()])],
                            batch[0][batch[0].nonzero()].astype(np.float32),
                            batch[0].shape
                        ),
                        dense_labels: batch[1]
                    }

                    val_err_batch, val_acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict)

                    val_err += val_err_batch
                    val_acc += val_acc_batch[0]
                    val_batches += 1

                val_time = time.time()
                print('Val {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, val_time - training_time))
                print('  val loss    :\t\t{:.6f}'.format(val_err / val_batches))
                print('  val accuracy:\t\t{:.2f} %'.format(val_acc / val_batches * 100))
                f.write('Val {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, val_time - training_time)+'\n')
                f.write('  val loss    :\t\t{:.6f}'.format(val_err / val_batches)+'\n')
                f.write('  val accuracy:\t\t{:.2f} %'.format(val_acc / val_batches * 100)+'\n')

                # Weights removal
                removed_weights = 0
                weights_total = 0
                for layer, values in kernels.items():
                    num_filter_values = sess.run(values['filter_values'])
                    current_small = np.abs(num_filter_values) < pruning_thr
                    weights_total += (num_filter_values != -1).sum()

                    if layer in to_remove:
                        prev_small = to_remove[layer]
                        num_filter_indices = sess.run(values['filter_indices'])
                        num_filter_channel_mapping = sess.run(values['filter_channel_mapping'])
                        indices_to_remove = prev_small & current_small
                        removed_weights += indices_to_remove.sum()

                        new_filter_values = num_filter_values[~indices_to_remove]
                        new_filter_indices = num_filter_indices[~indices_to_remove]
                        fill_values = -1 * np.ones(num_filter_values.shape[0] - new_filter_values.shape[0])
                        fill_indices = -1 * np.ones(num_filter_indices.shape[0] - new_filter_indices.shape[0])

                        to_subtract = np.zeros_like(num_filter_channel_mapping)
                        for i in range(1, len(num_filter_channel_mapping)):
                            to_subtract[i] = indices_to_remove[:num_filter_channel_mapping[i]].sum()

                        sess.run([
                            values['filter_channel_mapping'].assign(num_filter_channel_mapping - to_subtract),
                            values['filter_values'].assign(np.hstack([new_filter_values, fill_values])),
                            values['filter_indices'].assign(np.hstack([new_filter_indices, fill_indices]))
                        ])

                    to_remove[layer] = current_small

                print('Pruning {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, time.time() - val_time))
                print('  removed {} out of {} weights'.format(removed_weights, weights_total))
                f.write('Pruning {} of {} took {:.3f}s'.format(epoch + 1, num_epochs, time.time() - val_time)+'\n')
                f.write('  removed {} out of {} weights'.format(removed_weights, weights_total)+'\n')

            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=1)
            saver.save(sess, './' + '_thr_' + str(pruning_thr))

