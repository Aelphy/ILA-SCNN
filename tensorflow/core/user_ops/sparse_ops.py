from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import tensorflow as tf
from sparse_module import sparse_nn_ops as sparse_nn_ops

@ops.RegisterGradient("SparseTensorSparseKernelDenseConvKD")
def _SparseTensorSparseKernelDenseConvKDGrad(op, *grads):
  """Gradients for the SparseTensorSparseKernelDenseConvKD op.
  Args:
    op: the SparseTensorSparseKernelDenseConvKD op
    unused_output_indices_grad: the incoming gradients of the output indices
    output_values_grad: the incoming gradients of the output values
  Returns:
    Gradient for each of the 6 input tensors:
      (input_indices, input_values, input_shape, filter_indices, filter_value, filter_shape)
    The gradients for input_indices, input_shape, filter_indices and filter_shape are None.
  """
  return [None, 
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grads[1],
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None,
          None,
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_filter_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grads[1],
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None]

@ops.RegisterGradient("SparseTensorSparseKernelDenseConvKDInput")
def _SparseTensorSparseKernelDenseConvKDInputGrad(op, *grads):
  return [None,
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grads[1],
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None,
          None,
          None,
          None]


@ops.RegisterGradient("SparseTensorSparseKernelDenseConvKDFilter")
def _SparseTensorSparseKernelDenseConvKDFilterGrad(op, *grads):
  return [None,
          None,
          None,
          None,
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grads[1],
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None]



@ops.RegisterGradient("SparseRelu")
def _SparseReluGrad(op, *grads):
  return [None,
          sparse_nn_ops.sparse_relu_grad(op.inputs[0],
                                         op.inputs[1],
                                         op.outputs[1],
                                         grads[1],
                                         op.outputs[2]),
          None]


@ops.RegisterGradient("SparseFilterZeroOp")
def _SparseFilterZeroOpGrad(op, *grads):
  return [None,
          sparse_nn_ops.sparse_filter_zero_grad_op(op.inputs[0],
                                         op.outputs[0],
                                         grads[1]),
          None]

@ops.RegisterGradient("SparseTensorMaxPooling")
def _SparseTensorMaxPoolingGrad(op, *grads):
  return [None,
          sparse_nn_ops.sparse_tensor_max_pooling_grad(op.inputs[0],
                                         grads[1],
                                         op.outputs[3]),
          None,
          None]



@ops.RegisterGradient("DirectSparseToDense")
def _DirectSparseToDenseGrad(op, grad):
  return [None,
          None,
          tf.gather_nd(grad, op.inputs[0]),
          None]
