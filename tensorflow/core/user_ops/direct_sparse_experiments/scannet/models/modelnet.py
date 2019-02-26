import numpy as np
import tensorflow as tf
import direct_sparse_layer_definition as ld

def build_256(
    sparse_data, 
    tensor_in_sizes,
    num_classes=10,
    scope='sn256-',
    initializer=None,
    regularizer=None,
    d1=0.1,
    d2=0.15,
    d3=0.2
):
    dim = 5
    strides = [1,1,1,1,1]
    padding = 'SAME'
    pooling_sizes = [1,2,2,2,1]
    dpooling_sizes = [2,2,2]
    batch_size = tensor_in_sizes[0]
    total_size = np.prod(tensor_in_sizes)
    
    net = {}
    ops = [None]*9
    
    net['sd_converted'] = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
    net['conv1_1'], tensor_in_sizes, ops[0] = ld.create_sparse_conv_layer_reg(
        net['sd_converted'],
        [3,3,3,1,8],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d1,
        'K-ABS',
        scope + 'sc1',
        initializer
    )
    
    net['conv1_2'], tensor_in_sizes, ops[1] = ld.create_sparse_conv_layer_reg(
        net['conv1_1'],
        [3,3,3,8,8],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc2',
        initializer=initializer
    )
    net['conv1_3'], tensor_in_sizes, ops[2] = ld.create_sparse_conv_layer_reg(
        net['conv1_2'],
        [3,3,3,8,8],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc3',
        initializer=initializer
    )
    net['pool1'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv1_3'], pooling_sizes, tensor_in_sizes, dim, 0.1)
    net['conv2_1'], tensor_in_sizes, ops[3] = ld.create_sparse_conv_layer_reg(
        net['pool1'],
        [3,3,3,8,16],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d2,
        'K-ABS',
        name=scope + 'sc4',
        initializer=initializer
    )
    net['conv2_2'], tensor_in_sizes, ops[4] = ld.create_sparse_conv_layer_reg(
        net['conv2_1'],
        [3,3,3,16,16],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc5',
        initializer=initializer
    )
    net['conv2_3'], tensor_in_sizes, ops[5] = ld.create_sparse_conv_layer_reg(
        net['conv2_2'],
        [3,3,3,16,16],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc6',
        initializer=initializer
    )
    net['pool2'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv2_3'], pooling_sizes, tensor_in_sizes, dim, 0.18)
    net['conv3_1'], tensor_in_sizes, ops[6] = ld.create_sparse_conv_layer_reg(
        net['pool2'],
        [3,3,3,16,24],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d3, 
        'K-ABS',
        name=scope + 'sc7',
        initializer=initializer)
    net['conv3_2'], tensor_in_sizes, ops[7] = ld.create_sparse_conv_layer_reg(
        net['conv3_1'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides, padding,
        dim,
        d3,
        'K-RELU',
        name=scope + 'sc8',
        initializer=initializer)
    net['conv3_3'], tensor_in_sizes, ops[8] = ld.create_sparse_conv_layer_reg(
        net['conv3_2'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d3, 
        'K-RELU',
        name=scope + 'sc9',
        initializer=initializer)
    net['pool3'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv3_3'], pooling_sizes, tensor_in_sizes, dim, 0.5)
    net['sparse_to_dense'] = ld.create_direct_sparse_to_dense(net['pool3'], dim)
    net['dense_reshaped1'] = tf.reshape(net['sparse_to_dense'], [batch_size, 32, 32, 32, 24])
    net['conv4_1'] = tf.layers.conv3d(
        inputs=net['dense_reshaped1'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc10',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv4_2'] = tf.layers.conv3d(
        inputs=net['conv4_1'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc11',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv4_3'] = tf.layers.conv3d(
        inputs=net['conv4_2'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc12',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['pool4'] = tf.layers.max_pooling3d(inputs=net['conv4_3'], pool_size=dpooling_sizes, strides=2, padding='same', name='dp1')
    net['conv5_1'] = tf.layers.conv3d(
        inputs=net['pool4'],
        filters=40,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc13',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv5_2'] = tf.layers.conv3d(
        inputs=net['conv5_1'],
        filters=40,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc14',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv5_3'] = tf.layers.conv3d(
        inputs=net['conv5_2'],
        filters=40,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc15',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['pool5'] = tf.layers.max_pooling3d(inputs=net['conv5_3'], pool_size=dpooling_sizes, strides=2, padding='same', name='dp2')
    net['conv6_1'] = tf.layers.conv3d(
        inputs=net['pool5'],
        filters=48,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc16',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv6_2'] = tf.layers.conv3d(
        inputs=net['conv6_1'],
        filters=48,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name = scope + 'sc17',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv6_3'] = tf.layers.conv3d(
        inputs=net['conv6_2'],
        filters=48,
        kernel_size=[3, 3, 3],
        padding='same',
        name=scope + 'sc18')
    net['dense_reshaped2'] = tf.reshape(net['conv6_3'], [batch_size, -1])
    net['dense1'] = tf.layers.dense(net['dense_reshaped2'], 512)
    net['dense2'] = tf.layers.dense(net['dense1'], num_classes)

    return net, ops


def build_512(
    sparse_data, 
    tensor_in_sizes,
    num_classes=10,
    scope='sn256-',
    initializer=None,
    regularizer=None,
    d1=0.01,
    d2=0.03,
    d3=0.07
):
    dim = 5
    strides = [1,1,1,1,1]
    padding = 'SAME'
    pooling_sizes = [1,2,2,2,1]
    dpooling_sizes = [2,2,2]
    batch_size = tensor_in_sizes[0]
    total_size = np.prod(tensor_in_sizes)
    
    net = {}
    ops = [None]*12
    
    net['sd_converted'] = ld.create_sparse_data_to_direct_sparse(sparse_data, dim)
    net['conv1_1'], tensor_in_sizes, ops[0] = ld.create_sparse_conv_layer_reg(
        net['sd_converted'],
        [3,3,3,1,8],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d1,
        'K-ABS',
        scope + 'sc1',
        initializer
    )
    
    net['conv1_2'], tensor_in_sizes, ops[1] = ld.create_sparse_conv_layer_reg(
        net['conv1_1'],
        [3,3,3,8,8],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc2',
        initializer=initializer
    )
    net['conv1_3'], tensor_in_sizes, ops[2] = ld.create_sparse_conv_layer_reg(
        net['conv1_2'],
        [3,3,3,8,8],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d1,
        'K-RELU',
        name=scope + 'sc3',
        initializer=initializer
    )
    net['pool1'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv1_3'], pooling_sizes, tensor_in_sizes, dim, 0.06)
    net['conv2_1'], tensor_in_sizes, ops[3] = ld.create_sparse_conv_layer_reg(
        net['pool1'],
        [3,3,3,8,16],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d2,
        'K-ABS',
        name=scope + 'sc4',
        initializer=initializer
    )
    net['conv2_2'], tensor_in_sizes, ops[4] = ld.create_sparse_conv_layer_reg(
        net['conv2_1'],
        [3,3,3,16,16],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc5',
        initializer=initializer
    )
    net['conv2_3'], tensor_in_sizes, ops[5] = ld.create_sparse_conv_layer_reg(
        net['conv2_2'],
        [3,3,3,16,16],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d2,
        'K-RELU',
        name=scope + 'sc6',
        initializer=initializer
    )
    net['pool2'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv2_3'], pooling_sizes, tensor_in_sizes, dim, 0.18)
    net['conv3_1'], tensor_in_sizes, ops[6] = ld.create_sparse_conv_layer_reg(
        net['pool2'],
        [3,3,3,16,24],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d3, 
        'K-ABS',
        name=scope + 'sc7',
        initializer=initializer)
    net['conv3_2'], tensor_in_sizes, ops[7] = ld.create_sparse_conv_layer_reg(
        net['conv3_1'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides, padding,
        dim,
        d3,
        'K-RELU',
        name=scope + 'sc8',
        initializer=initializer)
    net['conv3_3'], tensor_in_sizes, ops[8] = ld.create_sparse_conv_layer_reg(
        net['conv3_2'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d3, 
        'K-RELU',
        name=scope + 'sc9',
        initializer=initializer)
    net['pool3'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv3_3'], pooling_sizes, tensor_in_sizes, dim, 0.5)
    net['conv4_1'], tensor_in_sizes, ops[9] = ld.create_sparse_conv_layer_reg(
        net['pool3'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d3, 
        'K-ABS',
        name=scope + 'sc10',
        initializer=initializer)
    net['conv4_2'], tensor_in_sizes, ops[10] = ld.create_sparse_conv_layer_reg(
        net['conv4_1'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides, padding,
        dim,
        d3,
        'K-RELU',
        name=scope + 'sc11',
        initializer=initializer)
    net['conv4_3'], tensor_in_sizes, ops[11] = ld.create_sparse_conv_layer_reg(
        net['conv4_2'],
        [3,3,3,24,24],
        tensor_in_sizes,
        strides,
        padding,
        dim,
        d3, 
        'K-RELU',
        name=scope + 'sc12',
        initializer=initializer)
    net['pool4'], tensor_in_sizes = ld.create_sparse_pooling_layer(net['conv4_3'], pooling_sizes, tensor_in_sizes, dim, 0.5)
    net['sparse_to_dense'] = ld.create_direct_sparse_to_dense(net['pool4'], dim)
    net['dense_reshaped1'] = tf.reshape(net['sparse_to_dense'], [batch_size, 32, 32, 32, 24])
    net['conv5_1'] = tf.layers.conv3d(
        inputs=net['dense_reshaped1'],
        filters=16,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc13',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv5_2'] = tf.layers.conv3d(
        inputs=net['conv5_1'],
        filters=16,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc14',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv5_3'] = tf.layers.conv3d(
        inputs=net['conv5_2'],
        filters=16,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc15',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['pool5'] = tf.layers.max_pooling3d(inputs=net['conv5_3'], pool_size=dpooling_sizes, strides=2, padding='same', name='dp1')
    net['conv6_1'] = tf.layers.conv3d(
        inputs=net['pool5'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc16',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv6_2'] = tf.layers.conv3d(
        inputs=net['conv6_1'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc17',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv6_3'] = tf.layers.conv3d(
        inputs=net['conv6_2'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc18',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['pool6'] = tf.layers.max_pooling3d(inputs=net['conv6_3'], pool_size=dpooling_sizes, strides=2, padding='same', name='dp2')
    net['conv7_1'] = tf.layers.conv3d(
        inputs=net['pool6'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name=scope + 'sc19',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv7_2'] = tf.layers.conv3d(
        inputs=net['conv7_1'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        activation=tf.nn.relu,
        name = scope + 'sc20',
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)
    net['conv7_3'] = tf.layers.conv3d(
        inputs=net['conv7_2'],
        filters=32,
        kernel_size=[3, 3, 3],
        padding='same',
        name=scope + 'sc21')
    net['pool7'] = tf.layers.max_pooling3d(inputs=net['conv7_3'], pool_size=dpooling_sizes, strides=2, padding='same', name='dp3')
    net['dense_reshaped2'] = tf.reshape(net['pool7'], [batch_size, -1])
    net['dense1'] = tf.layers.dense(net['dense_reshaped2'], 512)
    net['dense2'] = tf.layers.dense(net['dense1'], num_classes)

    return net, ops