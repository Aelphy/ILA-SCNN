from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

sparse_nn_ops = tf.load_op_library('sparse_tensor_dense_conv_3d.so')

@ops.RegisterGradient("SparseTensorSparseKernelDenseConvKD")
def _SparseTensorSparseKernelDenseConvKDGrad(op, grad):
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
  #input_indices = op.inputs[0]
  #input_values = op.inputs[1]
  #input_shape = op.inputs[2]
  #filter_indices = op.inputs[3]
  #filter_values = op.inputs[4]
  #filter_shape = op.inputs[5]
  #strides = op.get_attr("strides")
  #padding = op.get_attr("padding")


  return [None, 
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grad,
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None,
          None,
          #SparseTensorSparseKernelDenseConvKDFilterGrad
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_filter_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grad,
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None]

@ops.RegisterGradient("SparseTensorSparseKernelDenseConvKDInput")
def _SparseTensorSparseKernelDenseConvKDInputGrad(op, grad):
  return [None,
          sparse_nn_ops.sparse_tensor_sparse_kernel_dense_conv_kd_input_grad(op.inputs[0],
                                                op.inputs[1],
                                                op.inputs[2],
                                                op.inputs[3],
                                                op.inputs[4],
                                                op.inputs[5],
                                                op.outputs[0],
                                                grad,
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None,
          None,
          None,
          None]


@ops.RegisterGradient("SparseTensorSparseKernelDenseConvKDFilter")
def _SparseTensorSparseKernelDenseConvKDFilterGrad(op, grad):
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
                                                grad,
                                                op.outputs[2],
                                                strides=op.get_attr("strides"),
                                                padding=op.get_attr("padding")),
          None]
