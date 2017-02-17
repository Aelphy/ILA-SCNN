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

class SparseTensorSparseKernelDenseConv3DTest(test.TestCase):

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, expected):
    sc_module = tf.load_op_library('sparse_tensor_dense_conv_3d.so')
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s

    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session(use_gpu=True) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      t3 = tf.SparseTensor(indices=[[0, 1]],values=[1],dense_shape=[2,2]);
      sc_module.sparse_tensor_dense_conv3d.dense_to_sparse(t1, t3.indices, t3.values, t3.dense_shape,0);
      conv = nn_ops.conv3d(t1, t2, strides, padding="SAME")
      value = sess.run(conv)
    print("expected = ", expected)
    print("actual = ", value)
    expected = value.flatten() #TODO remove
    self.assertArrayNear(expected, value.flatten(), 1e-5)

  def testConv3D1x1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]

    # These are equivalent to the Conv2D1x1 case.
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 1, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 1, 2, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        expected=expected_output)

  # Expected values computed using scipy's correlate function.
  def testConv3D2x2x2Filter(self):
    expected_output = [
        19554., 19962., 20370., 22110., 22590., 23070., 34890., 35730., 36570.,
        37446., 38358., 39270., 50226., 51498., 52770., 52782., 54126., 55470.
    ]
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
        filter_in_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
        stride=1,
        expected=expected_output)

  def testConv3DStrides(self):
    expected_output = [
        102.,
        151.,
        172.,
        193.,
        214.,
        235.,
        142.,
        438.,
        592.,
        613.,
        634.,
        655.,
        676.,
        394.,
        774.,
        1033.,
        1054.,
        1075.,
        1096.,
        1117.,
        646.,
        1894.,
        2503.,
        2524.,
        2545.,
        2566.,
        2587.,
        1486.,
        2230.,
        2944.,
        2965.,
        2986.,
        3007.,
        3028.,
        1738.,
        2566.,
        3385.,
        3406.,
        3427.,
        3448.,
        3469.,
        1990.,
        3686.,
        4855.,
        4876.,
        4897.,
        4918.,
        4939.,
        2830.,
        4022.,
        5296.,
        5317.,
        5338.,
        5359.,
        5380.,
        3082.,
        4358.,
        5737.,
        5758.,
        5779.,
        5800.,
        5821.,
        3334.,
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 5, 8, 7, 1],
        filter_in_sizes=[1, 2, 3, 1, 1],
        stride=[2, 3, 1],  # different stride for each spatial dimension
        expected=expected_output)

  def testConv3D2x2x2FilterStride2(self):
    expected_output = [19554., 19962., 20370., 50226., 51498., 52770.]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        expected=expected_output)

  def testConv3DStride3(self):
    expected_output = [
        36564., 38022., 39480., 37824., 39354., 40884., 39084., 40686., 42288.,
        46644., 48678., 50712., 47904., 50010., 52116., 49164., 51342., 53520.,
        107124., 112614., 118104., 108384., 113946., 119508., 109644., 115278.,
        120912., 117204., 123270., 129336., 118464., 124602., 130740., 119724.,
        125934., 132144.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 6, 7, 8, 2],
        filter_in_sizes=[3, 2, 1, 2, 3],
        stride=3,
        expected=expected_output)

  def testConv3D2x2x2FilterStride2Same(self):
    expected_output = [
        19554., 19962., 20370., 10452., 10710., 10968., 50226., 51498., 52770.,
        23844., 24534., 25224.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        expected=expected_output)

  def testKernelSmallerThanStride(self):
    expected_output = [1., 3., 7., 9., 19., 21., 25., 27.]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        expected=expected_output)

    expected_output = [
        1484., 1592., 770., 2240., 2348., 1106., 1149., 1191., 539., 6776.,
        6884., 3122., 7532., 7640., 3458., 3207., 3249., 1421., 3005., 3035.,
        1225., 3215., 3245., 1309., 1013., 1022., 343.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        expected=expected_output)

    expected_output = [1484., 1592., 2240., 2348., 6776., 6884., 7532., 7640.]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        expected=expected_output)

  def testKernelSizeMatchesInputSize(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 2, 1],
        filter_in_sizes=[2, 1, 2, 1, 2],
        stride=1,
        expected=[50, 60])




if __name__ == "__main__":
  test.main()
