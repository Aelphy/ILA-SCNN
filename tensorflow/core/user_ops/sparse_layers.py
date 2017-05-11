from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.layers import base
from tensorflow.python.layers import utils


class SparseConv(base._Layer):  # pylint: disable=protected-access
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='Same',
               activation=None,
               kernel_initializer=None,
               kernel_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(SparseConv, self).__init__(trainable=trainable,
                                name=name, **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.activity_regularizer = activity_regularizer

  def build(self, input_shape):
    if len(input_shape) != self.rank + 2:
      raise ValueError('Inputs should have rank ' +
                       str(self.rank + 2) +
                       '. Received input shape: ' + str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = vs.get_variable('kernel',
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  trainable=True,
                                  dtype=self.dtype)
    if self.use_bias:
      self.bias = vs.get_variable('bias',
                                  shape=(self.filters,),
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  trainable=True,
                                  dtype=self.dtype)
    else:
      self.bias = None

  def call(self, inputs):
    outputs = nn.convolution(
        input=inputs,
        filter=self.kernel,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, self.rank + 2))
    if self.bias is not None:
      if self.rank != 2 and self.data_format == 'channels_first':
        # bias_add does not support channels_first for non-4D inputs.
        if self.rank == 1:
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        elif self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          outputs_4d = array_ops.reshape(outputs,
                                         [outputs_shape[0], outputs_shape[1],
                                          outputs_shape[2] * outputs_shape[3],
                                          outputs_shape[4]])
          outputs_4d = nn.bias_add(
              outputs_4d,
              self.bias,
              data_format=utils.convert_data_format(self.data_format, 4))
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(
            outputs,
            self.bias,
            data_format=utils.convert_data_format(self.data_format, 4))
        # Note that we passed rank=4 because bias_add will only accept
        # NHWC and NCWH even if the rank of the inputs is 3 or 5.

    if self.activation is not None:
      return self.activation(outputs)
    return outputs
