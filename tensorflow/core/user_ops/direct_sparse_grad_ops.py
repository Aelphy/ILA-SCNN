from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import tensorflow as tf
from direct_sparse_module import sparse_nn_ops as sparse_nn_ops

@ops.RegisterGradient("DirectSparseConvKD")
def _DirectSparseConvKDGrad(op, *grads):
  """Gradients for DirectSparseConvKD op.
  Args:
    op: the SparseTensorSparseKernelDenseConvKD op
    unused_output_indices_grad: the incoming gradients of the output indices
    output_values_grad: the incoming gradients of the output values
  Returns:
    Gradient for each of the 6 input tensors:
      (input_indices, input_values, input_shape, filter_indices, filter_value, filter_shape)
    The gradients for input_indices, input_shape, filter_indices and filter_shape are None.
  """
  [input_grads, filter_grads] = sparse_nn_ops.direct_sparse_conv_conv_kd_backprop(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.inputs[6],
                                                op.inputs[7],
                                                op.outputs[0],
                                                op.outputs[1],
                                                op.outputs[2],
                                                op.outputs[3],
                                                grads[1],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding"),
                                                dim=op.get_attr("dim"),
                                                max_density=op.get_attr("max_density"),
                                                filter_type=op.get_attr("filter_type")),
  return [None, 
          input_grads,
          None,
          None,
          None,
          filter_grads,
          None,
          None]

@ops.RegisterGradient("DirectSparseConvKDInput")
def _DirectSparseConvKDInputGrad(op, *grads):
  [input_grads, filter_grads] = sparse_nn_ops.direct_sparse_conv_conv_kd_backprop(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.inputs[6],
                                                op.inputs[7],
                                                op.outputs[0],
                                                op.outputs[1],
                                                op.outputs[2],
                                                op.outputs[3],
                                                grads[1],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding"),
                                                dim=op.get_attr("dim"),
                                                max_density=op.get_attr("max_density"),
                                                filter_type=op.get_attr("filter_type")),
  return [None,
          input_grads,
          None,
          None,
          None,
          None,
          None,
          None]


@ops.RegisterGradient("DirectSparseConvKDFilter")
def _DirectSparseConvKDFilterGrad(op, *grads):
  [input_grads, filter_grads] = sparse_nn_ops.direct_sparse_conv_conv_kd_backprop(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.inputs[6],
                                                op.inputs[7],
                                                op.outputs[0],
                                                op.outputs[1],
                                                op.outputs[2],
                                                op.outputs[3],
                                                grads[1],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding"),
                                                dim=op.get_attr("dim"),
                                                max_density=op.get_attr("max_density"),
                                                filter_type=op.get_attr("filter_type")),
  return [None,
          None,
          None,
          None,
          None,
          filter_grads,
          None,
          None]



@ops.RegisterGradient("DirectSparseMaxPoolingKD")
def _DirectSparseMaxPoolingKDGrad(op, *grads):
  return [None,
          sparse_nn_ops.direct_sparse_max_pooling_kd_backprop(op.inputs[0], 
                                         op.inputs[1], 
                                         op.inputs[2],
                                         op.inputs[3],
                                         op.outputs[0],
                                         op.outputs[1],
                                         op.outputs[2],
                                         op.outputs[3],
                                         grads[1],
                                         strides=op.get_attr("strides"),
                                         dim=op.get_attr("dim")),
          None,
          None]

@ops.RegisterGradient("DirectSparseUnpoolingKD")
def _DirectSparseUnpoolingKDGrad(op, *grads):
  return [None,
          sparse_nn_ops.direct_sparse_unpooling_kd_backprop(op.inputs[0], 
                                         op.inputs[1], 
                                         op.inputs[2],
                                         op.inputs[3],
                                         op.inputs[4],
                                         op.outputs[0],
                                         op.inputs[5],
                                         op.inputs[6],
                                         grads[1],
                                         strides=op.get_attr("strides"),
                                         dim=op.get_attr("dim")),
          None,
          None,
          None,
          None,
          None]


@ops.RegisterGradient("DirectSparseToDense")
def _DirectSparseToDenseGrad(op, grad):
  return [None,
          sparse_nn_ops.direct_sparse_to_dense_backprop(op.inputs[0],
                                         op.inputs[1],
                                         op.inputs[2],
                                         op.inputs[3],
                                         op.outputs[0],
                                         grad,
                                         dim=op.get_attr("dim")),
          None,
          None]

@ops.RegisterGradient("DirectDenseToSparse")
def _DirectDenseToSparseGrad(op, *grads):
  return sparse_nn_ops.direct_dense_to_sparse_backprop(op.inputs[0],
                                         op.outputs[0],
                                         op.outputs[1],
                                         op.outputs[2],
                                         op.outputs[3],
                                         grads[1],
                                         dim=op.get_attr("dim"))
